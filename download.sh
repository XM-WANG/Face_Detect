#!/bin/bash
mkdir model
wget -O model/age.prototxt https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/age.prototxt
wget -O model/dex_chalearn_iccv2015.caffemodel https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/dex_chalearn_iccv2015.caffemodel
wget -O model/gender.prototxt https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.prototxt
wget -O model/gender.caffemodel https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.caffemodel