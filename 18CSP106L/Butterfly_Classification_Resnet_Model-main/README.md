# butterfly-classification-resnet50

* Train, Test. Validation data set for 100 butterfly or moth species. All images are 224 X 224 X 3 in jpg format .
* Train set consists of 12639 images partitioned into 100 sub directories one for each species.
* Test set consists of 500 images partitioned into 100 sub directories with 5 test images per species.
* Valid set consists of 500 images partitioned into 100 sub directories with 5 validation images per species.
* A CSV files is included. The butterflies.csv file consists of 3 columns with 13639 rows, one row for each image in the dataset. The columns are filepaths, labels and data set. The filepaths column contains the relative path to the image.
* The labels column contains the species label associated with the image file. The data set column specifies which dataset (train, test or valid) the associated image belongs to.
