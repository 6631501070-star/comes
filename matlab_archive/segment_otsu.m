function [binaryImg, level] = segment_otsu(grayImg)
%SEGMENT_OTSU แปลงภาพเทาเป็นภาพไบนารีด้วยค่า threshold แบบ Otsu
%
%   เทคโนโลยี: วิธีของ Otsu เลือกค่าตัดที่ลดความแปรปรวนภายในกลุ่ม
%   (พื้นหน้า/พื้นหลัง) ให้ได้ผลดี ใช้เป็นทางเลือกหรือคู่ขนานกับการแยกมือ
%   สำหรับ fingerspelling A-Z เมื่อแสงค่อนข้างสม่ำเสมอ
%
%   รูปแบบการเรียกใช้:
%       [binaryImg, level] = segment_otsu(grayImg)
%
%   อินพุต:
%       grayImg - ภาพเทา (HxW) รองรับ uint8, uint16, int16, single หรือ double
%                 ถ้าส่งภาพ RGB (HxWx3) มาโดยไม่ตั้งใจ จะแปลงด้วย rgb2gray ให้
%
%   เอาต์พุต (ใช้กับ App Designer ได้):
%       binaryImg - ภาพไบนารี logical ขนาด HxW (true = พื้นหน้า)
%       level     - ค่า threshold แบบสเกลาร์ double ใน [0, 1] จาก graythresh
%
%   ตัวอย่าง (App Designer):
%       gray = rgb2gray(app.CurrentImage);
%       [bw, T] = segment_otsu(gray);
%       app.ThresholdEditField.Value = T;
%
%   Student 2 - Module 2: การแบ่งส่วนภาพ (Image Segmentation)

    %% -------------------- ตรวจสอบอินพุต --------------------
    if nargin < 1 || isempty(grayImg)
        error('segment_otsu:InvalidInput', ...
            'grayImg is required and must not be empty.');
    end

    if ~(isnumeric(grayImg) || islogical(grayImg))
        error('segment_otsu:InvalidType', ...
            'grayImg must be numeric or logical.');
    end

    %% -------------------- ให้ได้ภาพเทา 2 มิติ --------------------
    % ผู้ใช้ App Designer อาจส่ง RGB มาโดยไม่ตั้งใจ; แปลงอย่างปลอดภัย
    if ndims(grayImg) == 3 && size(grayImg, 3) == 3
        % แปลง RGB -> เทา (รองรับทั้ง uint8 และ double)
        if isinteger(grayImg)
            grayImg = rgb2gray(grayImg);
        else
            % rgb2gray ต้องการ double ในช่วง [0,1] สำหรับอินพุตแบบทศนิยม
            tmp = double(grayImg);
            if max(tmp(:)) > 1
                tmp = tmp / 255;
            end
            tmp = min(max(tmp, 0), 1);
            grayImg = rgb2gray(tmp);
        end
    elseif ~ismatrix(grayImg)
        error('segment_otsu:InvalidSize', ...
            'grayImg must be HxW grayscale or HxWx3 RGB.');
    end

    % ภาพ logical เป็นไบนารีอยู่แล้ว; ยังคืน level แบบง่ายไว้ให้ GUI
    if islogical(grayImg)
        binaryImg = grayImg;
        level = 0.5;
        return;
    end

    %% -------------------- ปรับชนิดข้อมูลให้เหมาะกับ graythresh --------------------
    % graythresh / imbinarize รองรับทั้งจำนวนเต็มและทศนิยม
    % ถ้าเป็น double ที่สเกล [0,255] ให้ปรับเป็น [0,1] ให้สอดคล้องกับ
    % มาตรฐานของ Image Processing Toolbox และการแสดงผลใน GUI
    workImg = grayImg;
    if isfloat(workImg)
        workImg = double(workImg);
        mx = max(workImg(:));
        if mx > 1
            workImg = workImg / 255;
        end
        workImg = min(max(workImg, 0), 1);
    elseif isinteger(workImg)
        % คงคลาสจำนวนเต็ม (เช่น uint8); graythresh จัดการผ่านฮิสโตแกรมภายใน
        % ไม่ต้องแปลงเพิ่มที่นี่
    else
        error('segment_otsu:UnsupportedType', ...
            'Unsupported image class: %s', class(workImg));
    end

    %% -------------------- คำนวณ threshold แบบ Otsu --------------------
    % เทคโนโลยี: graythresh ใช้วิธี Otsu (1979) — ค้นหาบนฮิสโตแกรมที่นอร์มัลไลซ์
    % เพื่อเพิ่มความแปรปรวนระหว่างกลุ่มให้มากที่สุด
    level = graythresh(workImg);

    %% -------------------- แปลงเป็นภาพไบนารี --------------------
    % imbinarize ใช้ค่า level ที่คำนวณได้; ผลลัพธ์เป็น logical (พร้อม GUI)
    % พื้นหน้า (มือ) อาจสว่างหรือมืดกว่าพื้นหลังตามแสง
    % ผู้เรียกสามารถกลับค่าด้วย ~binaryImg ได้ถ้าจำเป็น
    binaryImg = imbinarize(workImg, level);
end
