**Building a master reference of the pupal leg muscles**

**Terminology.** In muscle-GCaMP pupae, all brightly fluorescent tissue is muscle, but individual muscles are packed too closely to separate reliably.

- The *muscle mask* is the binary mask of all fluorescent muscle tissue.
- A *muscle cluster* is one connected region of that mask, usually several adjacent muscles.
- A *leg-segment mask* is the union of all clusters assigned to one leg segment.

1. **Volume preparation.**

   - Each recording (about 60 s, 46–53 volumes) was deskewed, and the per-voxel minimum over time (minimum projection) was taken as a resting-state image.
   - The volume was resampled to isotropic 1.5 µm voxels (linear interpolation).
   - The left–right mirror plane was found by maximising the normalised cross-correlation between the volume and its own reflection (Nelder–Mead on a 5× downsampled copy, optimising two tilt angles and an offset).
   - The volume was rotated so that this plane was perpendicular to one axis, split at the plane, and the right half was mirrored. This gave two hemibodies of the same handedness per animal: six hemibodies from three animals.

2. **Muscle masks.** Each hemibody was segmented automatically.

   - The log-transformed image (lightly smoothed, Gaussian σ = 1.5 µm) had its local background subtracted (Gaussian σ = 30 µm, computed over imaged voxels only) and was divided by its robust spread (1.4826 × the median absolute deviation, MAD).
   - This contrast image was thresholded with hysteresis (low 4, high 8 robust SDs).
   - Connected regions were removed if smaller than 5,000 µm³. Regions smaller than 30,000 µm³ were also removed unless elongated (ratio of the two largest principal axes ≥ 1.8) and at least 40 µm long, which rejects round blobs and specks.
   - Holes were filled slice-wise along each axis.
   - The remaining connected regions (face connectivity) are the muscle clusters.

3. **Reference leg segments.**

   - In one hemibody with complete hind-leg segments (the reference), each muscle cluster was assigned by hand to a leg segment: coxa/thorax, trochanter, femur and tibia of the front, middle and hind legs, or head.
   - This gives one leg-segment mask per segment.
   - The head was excluded from registration (see step 4).
   - No other hemibody needs segment labels.

4. **Registration.** Each hemibody (the target) was registered to the template by the shape of its muscle mask rather than by intensity. Leg muscles are long tubes whose intensity is nearly the same when they slide along their own axis, so the shape of the mask (its ends and outline) fixes their position better.

   - **Shape representation.** Each mask was represented as a signed distance map: the distance to the mask surface, negative inside and positive outside, clipped at ±30 µm and computed at 3 µm resolution.

   - **Cost function.** A fit minimises the mean squared difference between the template mask's distance map and the target's.

     - The difference is evaluated only inside the template mask. It therefore asks whether the target has muscle of matching thickness at each point of the template, and target muscle elsewhere (neighbouring segments, extra or unlabelled muscle) does not push the fit away.
     - Outside the target's field of view, the target's distance map is set to 0 ("unknown"). A segment cut off by the field of view is then matched by its visible part, at a small cost for the hidden part, and is neither pushed back into view nor able to escape out of it.

   - **Fitting steps:**

     1. *Initialisation:* a translation that superimposes the centroids of the template and target muscle masks.
     2. *Global fit:* the whole template muscle mask (all leg segments, head excluded) was fitted to the whole target muscle mask with a single affine transform (clip ±45 µm), starting from the initialisation. This corrects for differences in overall position, orientation and size between animals.
     3. *Segment fits:* each leg-segment mask of the template was fitted separately, starting from the global fit, against the target's whole muscle mask (all clusters, head included). No segment identities are used in the target. Segments larger than about 100,000 µm³ got a rigid fit (rotation about the segment's centroid plus translation) followed by an affine fit. Smaller segments (the trochanters and hind tibia) were fitted rigidly only. The segments of one leg can move relative to each other, and segment fits capture this.

     All fits used regular-step gradient descent over two resolution levels (6 and 3 µm), with up to 200 iterations per level (SimpleITK).

   - **Absent segments.** A segment fit was rejected, and the segment treated as absent from that target, when any of these held:

     - its centre moved more than 30 µm from where the global fit placed it;
     - less than 20% of it fell inside the target's field of view;
     - less than 30% of its visible part landed on target muscle (for example, a hind tibia outside the image).

5. **Deformation and resampling.**

   - The segment transforms of a target were combined into one smooth deformation. For each template voxel, the target positions given by all segment transforms were averaged, weighted by the distance to each template leg-segment mask (Gaussian, σ = 10 µm). The global transform carried a constant weight, equal to that of a segment 30 µm away, so it dominates far from all segments, including the head.
   - The target image and its muscle mask were sampled once at these positions (linear interpolation; mask threshold 0.5).
   - A target gave no data outside its field of view or within 20 µm of a segment absent from it.

6. **Averaging.**

   - Each image was normalised to 0 at its background (median of imaged voxels outside the muscle mask) and 1 at its muscle tissue (median inside).
   - The *master image* is the voxel-wise mean of the registered images, over the hemibodies with data at that voxel. The number of contributing hemibodies and the standard deviation were recorded per voxel.
   - The *master muscle mask* is a majority vote of the registered muscle masks: a voxel counts as muscle if at least half of the hemibodies with data there have muscle.
   - Each master-mask voxel within 10 µm of a template leg-segment mask takes that segment's identity. This gives the master's leg-segment masks; the head and other unassigned tissue stay unlabelled.

7. **Iteration.**

   - *Round 1:* the template was the reference hemibody's leg-segment masks. The other five hemibodies were registered to it, and the reference entered the average unchanged.
   - *Round 2:* the template was the round-1 master muscle mask with its leg-segment identities. All six hemibodies, the reference included, were registered to it, so the template shape is the consensus of all animals rather than one animal's.
   - Further rounds repeat round 2 with the previous round's master mask, until the results converge (three rounds here).
   - The last round's master image and master leg-segment masks form the master reference. New recordings are registered to it with steps 1, 2, 4 and 5, which need no manual labelling.
   - The coordinate frame (overall pose) remains that of the round-1 reference.

8. **Registering a new recording.**

   - A new recording is prepared and masked as in steps 1–2, and each of its hemibodies is registered to the master muscle mask (with its leg-segment identities) as in step 4.
   - To analyse the recording without deforming it, labels defined on the master (leg segments, or identified muscles) are brought back onto the recording instead. For each hemibody voxel, the master positions given by the segment transforms are averaged, with weights that fall off with the distance to each master segment as placed in the hemibody by its own transform (Gaussian, σ = 10 µm; the global transform takes over far from all segments). The label map is sampled there (nearest label).
   - The hemibody labels are then mapped back through the hemibody split (mirror, rotation and isotropic resampling, inverted) onto the voxel grid of the recording's original minimum projection.
