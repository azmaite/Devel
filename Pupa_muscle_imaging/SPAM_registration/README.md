## SPAM alignment pipeline

Pipeline designed to use the SPAM sofware (https://www.spam-project.dev/) to register 3D images of fly legs.

This is a challenging registration problem because each leg is more or less rigid but they can move relative to each other. The SPAM Discrete Image correlation package is made precisely to register rigid particles. It was developed (and recommended) by Edward Ando from the EPFL Imaging Processing hub.

This project will:
- Load a 3D image projection of fly legs, collected using SCAPE and already preprocessed and saved in an h5 file. Also load the voxel sizes, saved in the same h5 file.

- REMOVE THE ANISOTROPY - this is very important, as otherwise rotating a leg (or mirroring across a tilted plane) will make it look very different depending on the original angle in relation to the axes. Reduce resolution in some directions and increase it in the others to keep file size approximately similar. Remove the anisotropy on the segment labels and reference image too, keeping the same scale for both. This is done on the whole volumes, before finding the symmetry plane.

- Split the two hemibodies to process one at a time, by first finding the (slightly tilted) symmetry plane, rotating the volume so the plane is perpendicular to axis 2, and cropping. Flip one side to have two equivalent hemibodies

- Load the REFERENCE 3D projection, which has been manually segmented to isolate all the leg segments (femur, tibia, etc.). Split this reference too into two hemibodies

- Normalize the new image to match the reference image range

- Crop/pad the images to have identical dimensions for both.

- Replace the 'blank' background (either from the padding or from the scape deskewing) with random values matching the rest of the background.

- Perform a coarse, rigid alignment to get the two images approximately aligned - by then transfering the manual segmentation to the new image, this gives SPAM a decent starting point to find the matching segment in the new image. It is ESSENTIAL that there is some overlap between the new data and the manual segmentation, otherwise SPAM will not converge - this is why we do the pre-alignment.

- Run spam to align each segment one by one, using seed expansion = 0 (the manual segmentation already has some margins by design)