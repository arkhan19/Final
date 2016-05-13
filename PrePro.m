% Normalizing RGB image
clear all
Image_rgb = imread('Image/5.jpg');
Image_rgb = imresize(Image_rgb, [400 400]);
Image_rgb = double(Image_rgb);
%figure;imshow(uint8(Image_rgb));

Image_red = Image_rgb(:,:,1);
Image_green = Image_rgb(:,:,2);
Image_blue = Image_rgb(:,:,3);
%figure;imshow(uint8(Image_red));

[row,col] = size(Image_rgb(:,:,1));

for y = 1:row %-->numberof rows in image
   for x = 1:col  %-->number of columns in the image
      Red = Image_red(y,x);
      Green = Image_green(y,x);
      Blue = Image_blue(y,x);

    NormalizedRed = Red/sqrt(Red^2 + Green^2 + Blue^2);


    NormalizedGreen = Green/sqrt(Red^2 + Green^2 + Blue^2);
    NormalizedBlue = Blue/sqrt(Red^2 + Green^2 + Blue^2);
    Image_red(y,x) = NormalizedRed;
    Image_green(y,x) = NormalizedGreen;
    Image_blue(y,x) = NormalizedBlue;
   end
end

Image_rgb(:,:,1) = Image_red;
Image_rgb(:,:,2) = Image_green;
Image_rgb(:,:,3) = Image_blue;

Image_rgb = Image_rgb .* Image_rgb;
Image_rgb = Image_rgb .* Image_rgb;

figure; 
imshow(Image_rgb);
%imwrite (Image_rgb, 'Image/pout2.jpg');