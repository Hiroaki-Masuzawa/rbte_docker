mkdir images
# http://www.ess.ic.kanagawa-it.ac.jp/app_images_j.html
wget http://www.ess.ic.kanagawa-it.ac.jp/std_img/colorimage/color.zip -O images/color.zip
cd images && unzip -d color color.zip && rm color.zip