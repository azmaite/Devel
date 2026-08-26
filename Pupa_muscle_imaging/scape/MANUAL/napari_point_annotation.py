"""
NAPARI POINT ANNOTATION TOOL
Written manually with limited help from CLAUDE (max single functions)

Launches napari (open images/volumes as usual via File > Open) with:

- 'Other axes' (docked on the right): two extra linked views of the main
  viewer's layers, each sliced along a different pair of axes, so a 3D
  volume can be seen along all 3 axis orders at once. Napari's own
  'Change order of the visible axes' control (on the main view) drives
  all 3 views together. Unlike Grid mode - which tiles one view per
  *layer* - every layer (including a Points layer) is mirrored into both
  extra views, so points are overlaid on top of all 3 views rather than
  shown as a separate grid tile. Adapted from napari's official
  `multiple_viewer_widget.py` gallery example
  (https://napari.org/stable/gallery/multiple_viewer_widget.html),
  stripped down to just the linked views.

- A 'point label' row (plus Save/Load-to-JSON buttons), added to the
  Points layer's native controls panel (above the layer list, below the
  built-in point size/symbol/face color/border color rows). Create a
  Points layer (the 'New points layer' button in the layer list) and
  click in any of the 3 views to add points, in the layer's default 'Add
  points' mode - each new point gets a growing default label
  (1, 2, 3, ...). To rename one, switch the Points layer to 'Select
  points' mode (press `S`, or click the arrow icon above the layer list -
  clicking a point while still in 'Add points' mode adds a new point
  instead of selecting the existing one), then click the point (in any of
  the 3 views): its label appears in the 'point label' box, ready to edit
  (labels also render on-canvas, in all 3 views). Use 'Save to JSON' /
  'Load from JSON' to export/import a `{label: [z, y, x]}` dict, matching
  the point-name convention used in `align_acc_heart.ipynb` (e.g.
  `RF_femur_point1`).

launch_point_annotator: launch napari with the multi-axis-view widget attached and the point-label controls registered
_MultiAxisViewWidget: dock widget holding two extra linked views of the main viewer's layers, sliced along different axis orders
_QtLabelControl: adds a 'point label' edit box and save/load buttons to a Points layer's native controls panel
_QtPointsControlsWithLabel: `QtPointsControls` subclass with `_QtLabelControl` appended
_register_label_control: register `_QtPointsControlsWithLabel` as napari's Points layer controls widget
_copy_layer: copy a napari layer's data/state into a new, independent Layer instance
_own_partial: deepcopy-able stand-in for `functools.partial` (needed because Qt widgets referenced by a plain partial cannot be deep-copied)
_get_settable_property_names: get the names of a layer's settable properties that have a napari event to observe
"""

import json
from copy import deepcopy

import napari
import numpy as np
import pandas as pd
from napari._qt.layer_controls.qt_points_controls import QtPointsControls
from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase, QtWrappedLabel)
from napari.components.viewer_model import ViewerModel
from napari.layers import Layer, Points
from napari.qt import QtViewer
from napari.utils.events.event import WarningEmitter
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (QFileDialog, QHBoxLayout, QLineEdit, QPushButton,
                             QSplitter, QWidget)


def _copy_layer(layer: Layer, viewer_name: str) -> Layer:
    """
    Copy a napari layer's data/state into a new, independent Layer instance.

    Parameters
    ----------
    layer : napari.layers.Layer
        Layer to copy.

    viewer_name : str
        Name of the secondary viewer this copy belongs to (recorded in the
        copy's metadata, for debugging).

    Returns
    -------
    napari.layers.Layer
        A new layer instance with the same data/state as `layer`.
    """
    copied_layer = Layer.create(*layer.as_layer_data_tuple())
    copied_layer.metadata['viewer_name'] = viewer_name
    return copied_layer


class _own_partial:
    """
    Deepcopy-able stand-in for `functools.partial`.

    Needed because napari can deep-copy a layer's connected callbacks
    (e.g. when a layer is duplicated), and a plain `functools.partial`
    bound to a Qt widget method is not deep-copyable.

    Parameters
    ----------
    func : callable
        Function to call.

    *args, **kwargs
        Arguments to prepend/merge when `func` is called.
    """

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self.func(*(self.args + args), **{**self.kwargs, **kwargs})

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        return _own_partial(self.func, *deepcopy(self.args, memodict),
                             **deepcopy(self.kwargs, memodict))


def _get_settable_property_names(layer: Layer) -> list:
    """
    Get the names of a layer's settable properties that have a napari
    event to observe, for syncing property changes (other than data) to
    the layer copies in `_MultiAxisViewWidget`'s secondary views.

    Parameters
    ----------
    layer : napari.layers.Layer
        Layer to inspect.

    Returns
    -------
    list of str
        Names of properties with both a getter and a setter, excluding
        'thumbnail' and 'name' (synced separately).
    """
    layer_class = layer.__class__
    property_names = []
    for event_name, event_emitter in layer.events.emitters.items():
        if isinstance(event_emitter, WarningEmitter):
            continue
        if event_name in ('thumbnail', 'name'):
            continue
        class_attribute = getattr(layer_class, event_name, None)
        if isinstance(class_attribute, property) and class_attribute.fset is not None:
            property_names.append(event_name)
    return property_names


class _MultiAxisViewWidget(QSplitter):
    """
    Dock widget holding two extra views of the main viewer's layers, each
    sliced along a different pair of axes, so a 3D volume can be seen
    along all 3 axis orders at once.

    Every layer added to the main viewer - data, and settable properties
    such as a Points layer's `features` (i.e. its labels) - is mirrored
    into both secondary views, kept in sync in both directions (editing
    through any of the 3 views updates the other two).

    Parameters
    ----------
    viewer : napari.Viewer
        The main napari viewer to mirror.
    """

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.viewer_model1 = ViewerModel(title='model1')
        self.viewer_model2 = ViewerModel(title='model2')
        self._block = False
        # unlike napari's own layer events, psygnal's `Selection.events`
        # (used below for point-selection sync) only holds a *weak*
        # reference to connected callbacks, so an inline `_own_partial`
        # would be garbage-collected right away unless kept alive here
        self._selection_sync_callbacks = []

        self.qt_viewer1 = QtViewer(self.viewer_model1)
        self.qt_viewer2 = QtViewer(self.viewer_model2)
        self.setOrientation(Qt.Orientation.Vertical)
        self.addWidget(self.qt_viewer1)
        self.addWidget(self.qt_viewer2)

        self.viewer.layers.events.inserted.connect(self._on_layer_added)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)
        self.viewer.layers.events.moved.connect(self._on_layer_moved)
        self.viewer.layers.selection.events.active.connect(self._on_selection_changed)
        self.viewer.dims.events.current_step.connect(self._on_step_changed)
        self.viewer_model1.dims.events.current_step.connect(self._on_step_changed)
        self.viewer_model2.dims.events.current_step.connect(self._on_step_changed)
        self.viewer.dims.events.order.connect(self._on_order_changed)
        self.viewer.events.reset_view.connect(self._on_reset_view)
        self.viewer_model1.events.status.connect(self._on_status_changed)
        self.viewer_model2.events.status.connect(self._on_status_changed)

    def _on_status_changed(self, event):
        self.viewer.status = event.value

    def _on_reset_view(self, event=None):
        self.viewer_model1.reset_view()
        self.viewer_model2.reset_view()

    def _on_selection_changed(self, event):
        if self._block:
            return
        active_layer = event.value
        if active_layer is None:
            self.viewer_model1.layers.selection.active = None
            self.viewer_model2.layers.selection.active = None
        else:
            self.viewer_model1.layers.selection.active = self.viewer_model1.layers[active_layer.name]
            self.viewer_model2.layers.selection.active = self.viewer_model2.layers[active_layer.name]

    def _on_step_changed(self, event):
        for model in (self.viewer, self.viewer_model1, self.viewer_model2):
            if model.dims is event.source:
                continue
            if len(self.viewer.layers) != len(model.layers):
                continue
            model.dims.current_step = event.value

    def _on_order_changed(self, event=None):
        # each secondary view gets a different permutation of the last 3
        # axes, so all 3 orthogonal slicing planes are visible at once
        order = list(self.viewer.dims.order)
        if len(order) <= 2:
            self.viewer_model1.dims.order = order
            self.viewer_model2.dims.order = order
            return
        order[-3:] = order[-2], order[-3], order[-1]
        self.viewer_model1.dims.order = tuple(order)
        order = list(self.viewer.dims.order)
        order[-3:] = order[-1], order[-2], order[-3]
        self.viewer_model2.dims.order = tuple(order)

    def _on_layer_added(self, event):
        self.viewer_model1.layers.insert(event.index, _copy_layer(event.value, 'model1'))
        self.viewer_model2.layers.insert(event.index, _copy_layer(event.value, 'model2'))

        if isinstance(event.value, Points):
            # bake the label text display into every copy, since 'text' has
            # no napari event to pick up automatically (see _get_settable_property_names)
            points_layer_name = event.value.name
            role_to_model = {'main': self.viewer, 'model1': self.viewer_model1, 'model2': self.viewer_model2}
            for role, model in role_to_model.items():
                layer = model.layers[points_layer_name]
                layer.text = 'label'
                # so clicking a point in ANY of the 3 views (not just the main
                # one) is reflected in the shared point-label editor
                callback = _own_partial(self._on_point_selection_changed, points_layer_name, role)
                self._selection_sync_callbacks.append(callback)
                layer.selected_data.events.items_changed.connect(callback)

        for name in _get_settable_property_names(event.value):
            getattr(event.value.events, name).connect(_own_partial(self._on_property_changed, name))

        if event.value.name != '.cross':
            event.value.events.data.connect(self._on_data_changed)
            self.viewer_model1.layers[event.value.name].events.data.connect(self._on_data_changed)
            self.viewer_model2.layers[event.value.name].events.data.connect(self._on_data_changed)

        event.value.events.name.connect(self._on_name_changed)
        self._on_order_changed()

    def _on_layer_removed(self, event):
        self.viewer_model1.layers.pop(event.index)
        self.viewer_model2.layers.pop(event.index)

    def _on_layer_moved(self, event):
        destination_index = event.new_index if event.new_index < event.index else event.new_index + 1
        self.viewer_model1.layers.move(event.index, destination_index)
        self.viewer_model2.layers.move(event.index, destination_index)

    def _on_name_changed(self, event):
        index = self.viewer.layers.index(event.source)
        self.viewer_model1.layers[index].name = event.source.name
        self.viewer_model2.layers[index].name = event.source.name

    def _on_data_changed(self, event):
        if self._block:
            return
        for model in (self.viewer, self.viewer_model1, self.viewer_model2):
            layer = model.layers[event.source.name]
            if layer is event.source:
                continue
            try:
                self._block = True
                layer.data = event.source.data
            finally:
                self._block = False

    def _on_property_changed(self, name, event):
        if event.source not in self.viewer.layers:
            return
        try:
            self._block = True
            setattr(self.viewer_model1.layers[event.source.name], name, getattr(event.source, name))
            setattr(self.viewer_model2.layers[event.source.name], name, getattr(event.source, name))
        finally:
            self._block = False

    def _on_point_selection_changed(self, points_layer_name, source_role, *_args):
        # `Selection.events.items_changed` (unlike napari's own layer events)
        # emits plain positional args, not an Event with a `.source` - so
        # which of the 3 copies triggered this is bound at connect time
        # instead (see `_on_layer_added`)
        if self._block:
            return
        role_to_model = {'main': self.viewer, 'model1': self.viewer_model1, 'model2': self.viewer_model2}
        source_layer = role_to_model[source_role].layers[points_layer_name]
        try:
            self._block = True
            for role, model in role_to_model.items():
                if role == source_role:
                    continue
                model.layers[points_layer_name].selected_data = set(source_layer.selected_data)
        finally:
            self._block = False


class _QtLabelControl(QtWidgetControlsBase):
    """
    Adds a 'point label' edit box, plus Save/Load-to-JSON buttons, to a
    Points layer's native controls panel (appended below the built-in
    point size, symbol, face color, and border color rows - see
    `_QtPointsControlsWithLabel`).

    Labels are stored in the layer's `features['label']` column (and
    displayed on-canvas, via the layer's `text`). New points get a
    growing default label (1, 2, 3, ...); to rename one, switch the layer
    to 'Select points' mode (`S`) and click it - its label appears in the
    edit box, ready to edit.

    Parameters
    ----------
    parent : qtpy.QtWidgets.QWidget
        Parent widget (the owning `QtPointsControls` instance).

    layer : napari.layers.Points
        Points layer this control edits.
    """

    def __init__(self, parent, layer):
        super().__init__(parent, layer)
        self._known_length = len(self._layer.data)
        self._loading = False

        self._ensure_label_column()
        self._layer.text = 'label'
        self._next_label_number = self._compute_next_label_number()

        self.label_edit_label = QtWrappedLabel('point label:')
        self.label_edit = QLineEdit()
        self.label_edit.setEnabled(False)
        self.label_edit.editingFinished.connect(self._on_label_edited)

        self.save_button = QPushButton('Save to JSON')
        self.save_button.clicked.connect(self._on_save)
        self.load_button = QPushButton('Load from JSON')
        self.load_button.clicked.connect(self._on_load)
        self.save_load_label = QtWrappedLabel('labels file:')
        self.save_load_row = QWidget()
        save_load_layout = QHBoxLayout()
        save_load_layout.setContentsMargins(0, 0, 0, 0)
        save_load_layout.addWidget(self.save_button)
        save_load_layout.addWidget(self.load_button)
        self.save_load_row.setLayout(save_load_layout)

        self._layer.events.data.connect(self._on_data_changed)
        self._layer.selected_data.events.items_changed.connect(self._on_selection_changed)

        self._refresh_label_edit()

    def get_widget_controls(self):
        return [
            (self.label_edit_label, self.label_edit),
            (self.save_load_label, self.save_load_row),
        ]

    def _ensure_label_column(self):
        features = self._layer.features
        if 'label' not in features.columns:
            features = features.copy()
            features['label'] = [str(i + 1) for i in range(len(self._layer.data))]
            self._layer.features = features

    def _compute_next_label_number(self) -> int:
        labels = self._layer.features['label']
        numeric_labels = [int(label) for label in labels if str(label).isdigit()]
        return max(numeric_labels, default=0) + 1

    def _on_data_changed(self, event):
        if self._loading:
            return
        n_points = len(self._layer.data)
        previous_length = self._known_length
        self._known_length = n_points
        if n_points > previous_length:
            # a growing point count is relabeled regardless of the event's
            # `action` - a point added interactively fires action='added',
            # but a point added in one of the 2 secondary views arrives here
            # as a bulk `.data = ...` reassignment (via `_MultiAxisViewWidget`),
            # which fires action='changed' instead
            #
            # deferred to the next event-loop tick: reassigning `.features`
            # here fires a nested properties-changed event that
            # `_MultiAxisViewWidget` reacts to synchronously, and if that
            # ran *during* this same 'data' event (nested), the secondary
            # views' copies wouldn't have been resized to match yet -
            # deferring lets that data-sync finish first
            QTimer.singleShot(0, lambda: self._assign_default_labels(previous_length, n_points))
        self._refresh_label_edit()

    def _assign_default_labels(self, previous_length, n_points):
        if len(self._layer.data) != n_points:
            return  # data changed again before this ran; a newer call will handle it
        # napari already backfills new rows in `features` (e.g. with NaN) by
        # the time the 'data' event fires, so overwrite by index rather than
        # by comparing lengths (which already match at this point)
        features = self._layer.features.copy()
        labels = features['label'].tolist() if 'label' in features.columns else [''] * n_points
        for index in range(previous_length, n_points):
            labels[index] = str(self._next_label_number)
            self._next_label_number += 1
        features['label'] = labels
        self._layer.features = features
        self._refresh_label_edit()

    def _on_selection_changed(self, *_args):
        self._refresh_label_edit()

    def _refresh_label_edit(self):
        selected = list(self._layer.selected_data)
        if len(selected) == 1:
            label = self._layer.features['label'].iloc[selected[0]]
            self.label_edit.setEnabled(True)
            self.label_edit.setText(str(label))
        else:
            self.label_edit.setEnabled(False)
            self.label_edit.setText('')

    def _on_label_edited(self):
        selected = list(self._layer.selected_data)
        if len(selected) != 1:
            return
        features = self._layer.features.copy()
        labels = features['label'].tolist()
        labels[selected[0]] = self.label_edit.text()
        features['label'] = labels
        self._layer.features = features

    def _on_save(self):
        file_path, _ = QFileDialog.getSaveFileName(self.parent(), 'Save labels', '', 'JSON files (*.json)')
        if not file_path:
            return
        labels = self._layer.features['label'].astype(str).tolist()
        coords = self._layer.data.tolist()
        labeled_points = dict(zip(labels, coords))
        with open(file_path, 'w') as f:
            json.dump(labeled_points, f, indent=2)

    def _on_load(self):
        file_path, _ = QFileDialog.getOpenFileName(self.parent(), 'Load labels', '', 'JSON files (*.json)')
        if not file_path:
            return
        with open(file_path) as f:
            labeled_points = json.load(f)

        self._loading = True
        self._layer.data = np.array(list(labeled_points.values()))
        self._layer.features = pd.DataFrame({'label': list(labeled_points.keys())})
        self._loading = False

        self._known_length = len(self._layer.data)
        self._next_label_number = self._compute_next_label_number()
        self._refresh_label_edit()


class _QtPointsControlsWithLabel(QtPointsControls):
    """
    `QtPointsControls` with an extra 'point label' editing row (and
    Save/Load-to-JSON buttons) appended below the built-in point size,
    symbol, face color, and border color controls.

    Registered in place of napari's default Points controls widget by
    `_register_label_control`, which must run before any Points layer is
    added (see `launch_point_annotator`).

    Parameters
    ----------
    layer : napari.layers.Points
        An instance of a napari Points layer.
    """

    def __init__(self, layer):
        super().__init__(layer)
        self._label_control = _QtLabelControl(self, layer)
        self._add_widget_controls(self._label_control)


def _register_label_control():
    """
    Register `_QtPointsControlsWithLabel` in place of napari's default
    Points layer controls widget, so every Points layer's native controls
    panel gets the extra 'point label' editing row. Must run before any
    Points layer is added (controls widgets are created lazily, the first
    time a layer of that type is added/selected).
    """
    from napari._qt.layer_controls import qt_layer_controls_container
    qt_layer_controls_container.layer_to_controls[Points] = _QtPointsControlsWithLabel


def launch_point_annotator():
    """
    Launch napari with the multi-axis-view widget attached, and the
    point-label controls registered.

    Open an image/volume manually via napari's File > Open (or drag-and-
    drop) after launching. Create a Points layer to start labeling points:
    click to add points (each gets a growing default label); switch to
    'Select points' mode (`S`) and click a point to edit its label, in the
    'point label' row of the layer controls panel (above the layer list).
    """
    from qtpy import QtWidgets
    # needed to allow undocking the 'Other axes' widget (2 extra GL canvases)
    QtWidgets.QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

    _register_label_control()
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(_MultiAxisViewWidget(viewer), name='Other axes', area='right')
    napari.run()


if __name__ == '__main__':
    launch_point_annotator()
