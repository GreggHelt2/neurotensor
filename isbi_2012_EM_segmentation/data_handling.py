from PIL import Image, ImageSequence
import numpy as np
import glob
from imgaug import augmenters as iaa
import argparse
from os import makedirs
from os.path import basename, splitext, dirname

'''
General data handling and augmentation plan:
original data for ISBI 2012 EM segmentation challenge is in three mulit-image TIFF files: 
    each of the TIFF files is a stack of grayscale (L,uint8) images of 512 x 512 pixels, 
         with resolution 4nm x 4nm per pixel
         and z-direction inter-image resolution of 50 nm
         
    train-volume.tif: training data, images from serial section Transmission Electron Microscopy (ssTEM)
    label-volume.tif: per-pixel ground truth labeling for training data, 
                        full white (255) for pixels within segmented objects (presumably cells)
                        full black (0) for non-segmented pixels ("mostly membranes")
    test-volume.tif: test data, ssTEM similar to training data

Training data augmentation
1. split original multi-image TIFF files into separate file per image
2. for each training em/label pair, combine the two grayscale images into single RGB image,  
      with train as R channel and label as G channel
3. generate randomly rotated and deformed (rad_merged) new images from the merged training em/label images 
4. split rad_merged images into separate rad grayscale em and label images

Possible do 3 & 4 above on the fly at training time, for example:
      em, label = split_images(rad_aug(merged_images))  
      # em is list of EM images, label is list of per-pixel label images
      train(em, label, ....)
'''
# print("testing")
# original_datadir = "./original_data/"

def split_multi_tiff(source_file, dest_dir):
    print("-" * 50)
    makedirs(dest_dir, exist_ok=True)
    multi_tiff = Image.open(source_file)
    index = 0;
    for single_tiff in ImageSequence.Iterator(multi_tiff):
        if (index < 10):
            name = "0" + str(index)
        else:
            name = str(index)
        fname = dest_dir + "/" + name + ".tif"
        # print(fname)
        single_tiff.save(fname)
        index += 1
    print("split multi-image tiff:", source_file)
    print("split into", index, "single image tiff files in dir:" + dest_dir)
    print("-" * 50)

# expects each image in scandir to have a corresponding label image with same name in labeldir
def merge_images(scandir, labeldir, mergedir):
    print("-"*50)
    print("merging pairs of single-channel grayscale image from two directories into RGB images")
    print("image from", scandir, "becomes green channel of RGB result")
    print("image from", labeldir, "becomes blue channel of RGB result")
    makedirs(mergedir, exist_ok=True)
    scans = []
    labels = []
    merged = []
    names = []
    for path in sorted(glob.glob(scandir + "/*.tif")):
        base = basename(path)
        name, __ = splitext(base)
        # print(path)
        # print(base)
        scans.append(Image.open(path))
        lpath = labeldir + "/" + base
        labels.append(Image.open(lpath))
        names.append(name)
    #for name in glob.glob(labeldir + "/*.tif"):
    #    print(name)
    #    labels.append(Image.open(name))
    for index, scan_img in enumerate(scans):
        label_img = labels[index]
        nps = np.array(scan_img)
        npl = np.array(label_img)
        npm = np.zeros((512,512,3), dtype="uint8")
        npm[:,:,1] = nps
        npm[:,:,2] = npl
        merge_img = Image.fromarray(npm)
        merged.append(merge_img)
        name = names[index]
        merge_img.save(mergedir + "/" + name + ".tif")
        # merge_img.show(title="merge_"+str(index))
    print("created", len(merged), "merged images in:", mergedir)
    print("-"*50)
    return

def split_images(sourcedir, image_dest_dir, label_dest_dir):
    print("-" * 50)
    print("splitting merged images in:", sourcedir)
    print("  scan images saved in:", image_dest_dir)
    print("  label imaged saved in:", label_dest_dir)
    makedirs(image_dest_dir, exist_ok=True)
    makedirs(label_dest_dir, exist_ok=True)
    count = 0;
    for path in sorted(glob.glob(sourcedir + "/" + "/*.tif")):
        base = basename(path)
        name, __ = splitext(base)
        merged = Image.open(path)
        npm = np.array(merged)
        nps = np.zeros((512,512), dtype="uint8")
        npl = np.zeros((512,512), dtype="uint8")
        nps[:,:] = npm[:,:,1]
        npl[:,:] = npm[:,:,2]
        scan = Image.fromarray(nps)
        label = Image.fromarray(npl)
        count += 1
        scan.save(image_dest_dir + "/" + name + ".tif")
        label.save(label_dest_dir + "/" + name + ".tif")
    print("split", str(count), "merged train+label images")
    print("-" * 50)
    return

# rotation and deformation data augmentations
def augment_images(source_dir, dest_dir, cycles):
    print("-" * 50)
    print("making", cycles, "rotated and deformed versions of each image in:", source_dir)
    print("saving new images in: ", dest_dir)
    makedirs(dest_dir, exist_ok=True)
    rotate = iaa.Affine(rotate=(0, 360), mode='reflect')
    deform = iaa.PiecewiseAffine(scale=0.02, mode='reflect')
    total = 0
    for path in sorted(glob.glob(source_dir + "/*.tif")):
        orig = Image.open(path)
        base = basename(path)
        name, __ = splitext(base)
        norig = np.array(orig)
        # random rotation (with reflection to fill sided),
        # followed by piecewise affine deformations
        # I've had more success with imgaug's PiecewiseAffine than ElasticTransformation,
        #    and results from applytin PiecewiseAffine look to me more like the distortions achieved by the
        #    "elastic deformations" in Simard 2003 paper than results from applying ElasticTransformations

        print("  making rotated and deformed versions of:", path)
        for i in range(0,cycles):
            rad = deform.augment_image(rotate.augment_image(norig))
            new_image = Image.fromarray(rad)
            path = dest_dir + "/rad" + str(i) + "_" + name + ".tif"
            new_image.save(path)
            total += 1
    print("total new images:", total)
    print("-" * 50)
    return

# save all tiff images in a directorty as single numpy array: (IMAGE_COUNT, 512, 512, 1)
# assumes images are single-channel (grayscale) and same width and height
def save_as_numpy(source_dir, dest_path):
    # makedirs(dest_dir, exist_ok=True)
    makedirs(dirname(dest_path), exist_ok=True)
    imgs = glob.glob(source_dir + "/*.tif")
    icnt = len(imgs)
    print("number of images:", icnt)
    index = 0
    all_images = np.ndarray((icnt, 512, 512, 1), dtype="uint8")
    for path in sorted(imgs):
        print(path)
        img = Image.open(path)
        npimg = np.array(img)
        npimg = npimg.reshape(512,512,1)
        all_images[index] = npimg
        index += 1
    np.save(dest_path, all_images)
    print("saved all tiff images in:", source_dir, "as single numpy array in:", dest_path)
    return

def split_original_data():
    split_multi_tiff("./data_original/train-volume.tif", "./data_original_split/train/images")
    split_multi_tiff("./data_original/train-labels.tif", "./data_original_split/train/labels")
    split_multi_tiff("./data_original/test-volume.tif", "./data_original_split/test/images")

# resulting data structure:
#     dest_root
#         info.txt (info on source_root, augment_training, validation_fraction)
#         training
#             merged
#             merged_augmented (includes merged originals?)
#             images_augmented (includes split originals)
#             labels_augmented (includes split originals)
#             train_images.npy
#             train_labels.npy
#         validation
#             images
#             labels
#             validation_images.npy
#             validation_labels.npy
#         test
#             images
#             test_images.npy
# possibly don't need the npy arrays above (not sure how much they improve performance)
# augment_training = 0 ==> no augmentation
def make_train_and_validation_data(source_root, dest_root, augment_training=10, validation_fraction=0.2):
    return

# dest_root
#     training_params.hdf5
#     training_stats.txt
def train(dest_root):
    return

# dest_root
#     prediction_stats.txt
#     predicted_labels
#         [label_images.tif]*
#     predicted_labels.tif (multi-image tif from stack of all prediction images)
def predict():
    return

if __name__ == "__main__":
    print("running data_handling __main__")
    parser = argparse.ArgumentParser(description='ISBI 2012 data handling and augmentation')
    argtest = parser.parse_args()
    # split_original_data()
    # merge_images("./data_original_split/train/images", "./data_original_split/train/labels", "./data_training_merged")
    # augment_images("./data_training_merged", "./data_training_merged_augmented", 10)
    # split_images("./data_training_merged_augmented", "./data_training_augmented/training/images","./data_training_augmented/training/labels")
    # save_as_numpy("./augmented_and_original_data/training/images", "./augor_numpy_data/train_images.npy")
    # save_as_numpy("./augmented_and_original_data/training/labels", "./augor_numpy_data/train_labels.npy")
    save_as_numpy("./data_original_split/test/images", "./augor_numpy_data/test_images.npy")

