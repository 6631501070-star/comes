function [binaryMask, segmentedImg] = segment_skin_hsv(rgbImg, hsvLimits)
%SEGMENT_SKIN_HSV แยกบริเวณมือ/ผิวหนังด้วยการกำหนดช่วงสีในระบบ HSV
%
%   เทคโนโลยี: ใช้การแบ่งส่วนภาพด้วยสีในปริภูมิ HSV สำหรับ fingerspelling (A-Z)
%   โดยกำหนดช่วง Hue, Saturation และ Value เพื่อแยกมือออกจากพื้นหลัง
%   ให้โมดูลถัดไปดึงลักษณะรูปร่างนิ้วไปจำแนกตัวอักษรได้
%
%   รูปแบบการเรียกใช้:
%       [binaryMask, segmentedImg] = segment_skin_hsv(rgbImg)
%       [binaryMask, segmentedImg] = segment_skin_hsv(rgbImg, hsvLimits)
%
%   อินพุต:
%       rgbImg    - ภาพ RGB (uint8 หรือ double) ถ้าเป็น double ควรอยู่ในช่วง [0, 1]
%                   (ใช้ร่วมกับ MATLAB Image Processing Toolbox)
%       hsvLimits - struct แบบเลือกใส่ สำหรับผูกกับ App Designer:
%                   .Hmin, .Hmax  - ช่วง Hue ใน [0, 1]
%                   .Smin, .Smax  - ช่วง Saturation ใน [0, 1]
%                   .Vmin, .Vmax  - ช่วง Value (ความสว่าง) ใน [0, 1]
%                   ถ้าไม่ใส่ จะใช้ช่วงสีผิวเริ่มต้นสำหรับแสงในร่มทั่วไป
%
%   เอาต์พุต (ใช้กับ GUI ได้สะดวก):
%       binaryMask   - มาสก์ logical ขนาด HxW (true = พิกเซลผิว/มือ)
%       segmentedImg - ภาพ RGB คลาสเดียวกับอินพุต; พื้นหลังที่ไม่ใช่ผิวถูกตั้งเป็น 0
%
%   ตัวอย่าง (ใน callback ของ App Designer):
%       limits.Hmin = 0; limits.Hmax = 0.1;
%       limits.Smin = 0.2; limits.Smax = 0.7;
%       limits.Vmin = 0.2; limits.Vmax = 1.0;
%       [mask, skin] = segment_skin_hsv(app.CurrentImage, limits);
%
%   Student 2 - Module 2: การแบ่งส่วนภาพ (Image Segmentation)

    %% -------------------- ตรวจสอบอินพุต --------------------
    % ต้องมีภาพส่งเข้ามา
    if nargin < 1 || isempty(rgbImg)
        error('segment_skin_hsv:InvalidInput', ...
            'rgbImg is required and must not be empty.');
    end

    % ภาพ RGB ต้องมี 3 ช่องสี (H x W x 3)
    if ndims(rgbImg) ~= 3 || size(rgbImg, 3) ~= 3
        error('segment_skin_hsv:InvalidInput', ...
            'rgbImg must be an RGB image of size HxWx3.');
    end

    % รองรับชนิดที่พบบ่อยจาก App Designer / imread: uint8 และ double
    if ~(isinteger(rgbImg) || isfloat(rgbImg))
        error('segment_skin_hsv:InvalidType', ...
            'rgbImg must be an integer or floating-point array (e.g. uint8 or double).');
    end

    %% -------------------- ปรับค่าเป็น double ในช่วง [0, 1] --------------------
    % rgb2hsv ต้องการ double ในช่วง [0, 1] เพื่อให้ช่วง HSV ถูกต้อง
    % เก็บต้นฉบับไว้สร้าง segmentedImg ให้คลาสตรงกับอินพุต (สำคัญตอนโชว์ใน GUI)
    originalImg = rgbImg;
    if isinteger(rgbImg)
        % แปลงภาพจำนวนเต็ม (มักเป็น uint8 [0,255]) ให้เป็น [0,1]
        rgbDouble = im2double(rgbImg);
    else
        rgbDouble = double(rgbImg);
        % ถ้าผู้เรียกส่ง double ในช่วง [0,255] ให้หารกลับเป็น [0,1]
        if max(rgbDouble(:)) > 1
            rgbDouble = rgbDouble / 255;
        end
        % ตัดค่าที่อยู่นอกช่วงเพื่อความปลอดภัยเชิงตัวเลข
        rgbDouble = min(max(rgbDouble, 0), 1);
    end

    %% -------------------- ช่วงสีผิวเริ่มต้น / ที่ผู้ใช้กำหนด --------------------
    % ค่าเริ่มต้นเหมาะกับสีผิวทั่วไปภายใต้แสงในร่ม
    % Hue วนรอบสีแดง (ใกล้ 0 และใกล้ 1) จึงใช้สองช่วง Hue
    % App Designer สามารถทับค่าฟิลด์ใดก็ได้ผ่าน hsvLimits
    defaults.Hmin = 0.00;
    defaults.Hmax = 0.10;   % ช่วงแดง/ส้มด้านล่าง
    defaults.Hmin2 = 0.90;  % ช่วงแดงที่วนกลับด้านบน
    defaults.Hmax2 = 1.00;
    defaults.Smin = 0.20;
    defaults.Smax = 0.70;
    defaults.Vmin = 0.20;
    defaults.Vmax = 1.00;

    if nargin < 2 || isempty(hsvLimits)
        hsvLimits = defaults;
    else
        if ~isstruct(hsvLimits)
            error('segment_skin_hsv:InvalidLimits', ...
                'hsvLimits must be a struct with fields Hmin, Hmax, Smin, Smax, Vmin, Vmax.');
        end
        % รวมค่าผู้ใช้ทับค่าเริ่มต้น (ฟิลด์ที่ขาดให้ใช้ค่าเริ่มต้น)
        fn = fieldnames(defaults);
        for i = 1:numel(fn)
            name = fn{i};
            if ~isfield(hsvLimits, name) || isempty(hsvLimits.(name))
                hsvLimits.(name) = defaults.(name);
            end
        end
    end

    %% -------------------- แปลง RGB -> HSV --------------------
    % เทคโนโลยี: RGB-to-HSV แยกข้อมูลสี (H,S) ออกจากความสว่าง (V)
    % ทำให้จับสีผิวทนต่อแสงเปลี่ยนแปลงได้ดีกว่าใช้ RGB อย่างเดียว
    hsvImg = rgb2hsv(rgbDouble);
    H = hsvImg(:, :, 1);  % Hue        [0, 1]
    S = hsvImg(:, :, 2);  % Saturation [0, 1]
    V = hsvImg(:, :, 3);  % Value      [0, 1]

    %% -------------------- สร้างมาสก์ผิวแบบไบนารี --------------------
    % Hue ใกล้แดงอาจวนรอบ: รับได้ทั้งช่วงต่ำ หรือช่วงสูง
    hueMask = ((H >= hsvLimits.Hmin) & (H <= hsvLimits.Hmax)) | ...
              ((H >= hsvLimits.Hmin2) & (H <= hsvLimits.Hmax2));

    satMask = (S >= hsvLimits.Smin) & (S <= hsvLimits.Smax);
    valMask = (V >= hsvLimits.Vmin) & (V <= hsvLimits.Vmax);

    % มาสก์สุดท้าย: true ที่พิกเซลน่าจะเป็นผิว/มือ
    binaryMask = hueMask & satMask & valMask;

    %% -------------------- สร้างภาพ RGB ที่ตัดพื้นหลังแล้ว --------------------
    % คงพิกเซลผิวเดิมไว้; ตั้งพื้นหลังเป็น 0 สำหรับพรีวิวใน GUI
    segmentedImg = originalImg;
    % ทำซ้ำมาสก์ให้ครบช่อง R,G,B
    mask3 = repmat(binaryMask, [1, 1, 3]);
    segmentedImg(~mask3) = 0;
end
