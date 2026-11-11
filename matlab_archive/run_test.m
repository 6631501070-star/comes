%% run_test.m
% ทดสอบ Module 2 แบบเร็ว: แสดงผลทีละรูป และบันทึกทุก Figure ลง .\results\
%
% วิธีใช้:
%   cd('D:\Coding\ComVision')
%   run_test

clear; close all; clc;

%% การตั้งค่า
inputImage = 'test_hand.jpg';
outDir     = fullfile(pwd, 'results');
imgFormat  = 'png';

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

%% 1) ภาพต้นฉบับ
rgb = imread(inputImage);
figSave(rgb, outDir, '01_original_rgb', 'Original RGB', imgFormat);

%% 2) การแยกผิวด้วย HSV
lim.Hmin = 0;    lim.Hmax = 0.12;
lim.Smin = 0.15; lim.Smax = 0.75;
lim.Vmin = 0.15; lim.Vmax = 1;

[mask, skin] = segment_skin_hsv(rgb, lim);
figSave(mask, outDir, '02_skin_binary_mask', 'Skin binary mask', imgFormat);
figSave(skin, outDir, '03_segmented_skin', 'Segmented skin', imgFormat);

%% 3) Otsu
gray = rgb2gray(rgb);
[bw, level] = segment_otsu(gray);
figSave(bw, outDir, '04_otsu', sprintf('Otsu, level=%g', level), imgFormat);
% ถ้ามือเป็นสีดำ ให้ลอง: bw = ~bw;

%% 4) Morphology
R = apply_morphology(mask, 'disk', 5);
figSave(R.cleanedMask, outDir, '05_cleaned_mask', 'Cleaned mask', imgFormat);
% ลองเทียบ: R.openedMask, R.closedMask, R.filledMask

%% 5) การตรวจจับขอบ
E = detect_edges_all(R.cleanedMask);  % หรือ detect_edges_all(gray)
figSave(E.sobelEdges,   outDir, '06_sobel',   'Sobel', imgFormat);
figSave(E.prewittEdges, outDir, '07_prewitt', 'Prewitt', imgFormat);
figSave(E.cannyEdges,   outDir, '08_canny',   'Canny', imgFormat);

fprintf('\nAll figures saved in:\n  %s\n', outDir);

%% -------- ฟังก์ชันช่วยในไฟล์ --------
function figSave(img, outDir, fileStem, figTitle, imgFormat)
    % แสดงภาพในหน้าต่าง Figure แล้วบันทึกลงดิสก์
    figure('Name', figTitle, 'NumberTitle', 'off', 'Color', 'w');
    imshow(img);
    title(figTitle, 'Interpreter', 'none');
    drawnow;

    outPath = fullfile(outDir, [fileStem '.' imgFormat]);
    try
        exportgraphics(gcf, outPath, 'Resolution', 150);
    catch
        saveas(gcf, outPath);
    end
    fprintf('Saved: %s\n', outPath);
end
