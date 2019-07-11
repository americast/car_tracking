# sudo apt install libeigen3-dev
export CPATH="$CPATH:$(pwd)/json/include:$(pwd)/FunctionalPlus/include:$(pwd)/frugally-deep/include"
g++ -I/usr/include/eigen3/ in_cpp.cc


