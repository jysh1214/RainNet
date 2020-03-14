# Learning to Group Discrete Graphical Patterns

----------

[Project Page](http://people.cs.umass.edu/~zlun/papers/PatternGrouping/)

## Introduction

This archive contains **Training Data** used in our experiments. Data are organized in Matlab's MAT-file format.

## Data Format

The file `list.txt` contains a list of names for all 7891 training cases. Below are the descriptions for files within each corresponding sub-folders. Each file corresponds to one case storing in Matlab's MAT-file format (except for the image files which are in PNG format).

### element

The data field `eleIDMatrix` is a 800x800 matrix indicating the element ID for each pixel. The element ID for background region pixels is 0. 

### label

The data field `labelingMatrix` is a 800x800 matrix indicating the ground-truth label for each pixel. Pixels within the same element (having the same element ID) should have the same label. The label for background region pixels is 0.

### image

The black-and-white PNG image visualizes the pattern image (background region is colored in white).

## Other Notes

- Our data is provided for academic use only.
- If you would like to use our data, please cite the following paper:

	> Zhaoliang Lun*, Changqing Zou*, Haibin Huang, Evangelos Kalogerakis, Ping Tan, Marie-Paule Cani, Hao Zhang,
	"Learning to Group Discrete Graphical Patterns",
	ACM Transactions on Graphics, Vol. 36, No. 10, 2017 (Proc. of ACM SIGGRAPH ASIA 2017)


- For any questions or comments, please contact Changqing Zou ([aaronzou1125@gmail.com](mailto:aaronzou1125@gmail.com))