# YouTube8M_tools
Convert feature.pb to yt8m style tfrecord

1. This is a simple tool for data transformation of mediapipe feature.pb to yt8m tfrecord.
2. The version of mediapipe is 0.7.4, tensorflow=1.14.
3. Source data of this tool is feature.pb extracted from local video including auio&rgb features.

Kindly contact if any quesitons. lol

###################### updated 2020/06/23 ##########################

Newly added one script that can run under docker container mediapipe.

It helps you convert multiple files into single tfrecord.

p.s. now only works for video with audio, I'm working on script works for video without audio.

###################### updated 2020/06/29 ##########################

Now this script can process videos with or without audio in it.

Hint: need to modify the absolute path of bazel-bin and calculator_graph_config_file before you run the script.
