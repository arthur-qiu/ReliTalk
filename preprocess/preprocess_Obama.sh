#######################################################
# Things you need to modify
subject_name='Obama'
path='../data/datasets'
video_folder=$path/$subject_name
video_names='Obama_train.mp4 Obama_eval.mp4'
shape_video='Obama_train.mp4'
fps=25
# Center crop
crop="450:450:0:0"
resize=256
# fx, fy, cx, cy in pixels, need to adjust with resizing and cropping
fx=1200.0
fy=1200.0
cx=225.0
cy=225.0
########################################################
pwd=$(pwd)
path_modnet=$(pwd)'/submodules/MODNet'
path_deca=$(pwd)'/submodules/DECA'
path_parser=$(pwd)'/submodules/face-parsing.PyTorch'
########################################################
set -e
echo "crop and resize video"
cd $pwd
for video in $video_names
do
  video_path=$video_folder/$video
  echo $video
  IFS='.' read -r -a array <<< $video
  echo $video_folder/$subject_name/"${array[0]}"/"image"
  ffmpeg -y -i $video_path -vf "fps=$fps, crop=$crop, scale=$resize:$resize" -c:v libx264 $video_folder/"${array[0]}_cropped.mp4"
done
echo "background/foreground segmentation"
cd $path_modnet
for video in $video_names
do
  video_path=$video_folder/$video
  echo $video
  IFS='.' read -r -a array <<< $video
  mkdir -p $video_folder/$subject_name/"${array[0]}"
  python -m demo.video_matting.custom.run --video $video_folder/"${array[0]}_cropped.mp4" --result-type matte --fps $fps
done
echo "save the images and masks with ffmpeg"
# sudo apt install ffmpeg
cd $pwd
for video in $video_names
do
  video_path=$video_folder/$video
  echo $video
  IFS='.' read -r -a array <<< $video
  echo $video_folder/$subject_name/"${array[0]}"/"image"
  mkdir -p $video_folder/$subject_name/"${array[0]}"/"image"
  ffmpeg -i $video_folder/"${array[0]}_cropped.mp4" -q:v 2 $video_folder/$subject_name/"${array[0]}"/"image"/"%d.png"
  mkdir -p $video_folder/$subject_name/"${array[0]}"/"mask"
  ffmpeg -i $video_folder/"${array[0]}_cropped_matte.mp4" -q:v 2 $video_folder/$subject_name/"${array[0]}"/"mask"/"%d.png"
done
echo "DECA FLAME parameter estimation"
cd $path_deca
for video in $video_names
do
  echo $video
  IFS='.' read -r -a array <<< $video
  mkdir -p $video_folder/$subject_name/"${array[0]}"/"deca"
  python demos/demo_reconstruct.py -i $video_folder/$subject_name/"${array[0]}"/image --savefolder $video_folder/$subject_name/"${array[0]}"/"deca" --saveCode True --saveVis False --sample_step 1  --render_orig False
done
echo "face alignment landmark detector"
cd $pwd
for video in $video_names
do
  echo $video
  IFS='.' read -r -a array <<< $video
  python keypoint_detector.py --path $video_folder/$subject_name/"${array[0]}"
done
echo "iris segmentation with fdlite"
cd $pwd
for video in $video_names
do
  video_path=$video_folder/$video
  echo $video
  IFS='.' read -r -a array <<< $video
  python iris.py --path $video_folder/$subject_name/"${array}"
done
echo "fit FLAME parameter for one video: "$shape_video
cd $path_deca
IFS='.' read -r -a array <<< $shape_video
python optimize.py --path $video_folder/$subject_name/"${array}" --cx $cx --cy $cy --fx $fx --fy $fy --size $resize
echo "fit FLAME parameter for other videos, while keeping shape parameter fixed"
cd $path_deca
for video in $video_names
do
  if [ "$shape_video" == "$video" ];
  then
    continue
  fi
  IFS='.' read -r -a array <<< $(basename $shape_video)
  shape_from=$video_folder/$subject_name/"${array}"
  IFS='.' read -r -a array <<< $(basename $video)
  echo $video
  python optimize.py --path $video_folder/$subject_name/"${array}" --shape_from $shape_from  --cx $cx --cy $cy --fx $fx --fy $fy --size $resize
done
echo "semantic segmentation with face parsing"
cd $path_parser
for video in $video_names
do
  video_path=$video_folder/$video
  echo $video
  IFS='.' read -r -a array <<< $video
  python test.py --dspth $video_folder/$subject_name/"${array}"/image --respth $video_folder/$subject_name/"${array}"/semantic
done
