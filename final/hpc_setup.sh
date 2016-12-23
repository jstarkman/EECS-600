echo Usage: $ source hpc_setup.sh
echo Do not execute.
module load caffe/2015
module load python/3.5.2
echo 'Assumes a local version of Caffe ($ cp -r $CAFFE_ROOT ~/caffe/)'

wget https://github.com/ZhiangChen/deep_learning/raw/master/27_objects/lenet5/depth_data

