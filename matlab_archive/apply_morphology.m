function morphResults = apply_morphology(binaryMask, seShape, seSize)
%APPLY_MORPHOLOGY ทำความสะอาดมาสก์มือแบบไบนารีด้วยปฏิบัติการทางสัณฐานวิทยา
%
%   เทคโนโลยี: Mathematical morphology ใช้ structuring element (SE)
%   เพื่อสำรวจและปรับรูปร่างบริเวณไบนารี สำหรับ fingerspelling A-Z
%   - Opening ช่วยลบจุดรบกวนขนาดเล็ก
%   - Closing ช่วยอุดช่องว่างแคบระหว่างนิ้ว/ฝ่ามือ
%   - Erosion / Dilation ปรับความหนาของขอบ
%   - Hole filling อุดรูภายในบริเวณมือ
%
%   รูปแบบการเรียกใช้:
%       morphResults = apply_morphology(binaryMask)
%       morphResults = apply_morphology(binaryMask, seShape, seSize)
%
%   อินพุต:
%       binaryMask - ภาพไบนารี logical หรือตัวเลข (HxW)
%                    ถ้าเป็นตัวเลขจะแปลงด้วย logical(binaryMask ~= 0)
%       seShape    - รูปทรงของ SE สำหรับ strel (ค่าเริ่มต้น: 'disk')
%                    ที่พบบ่อย: 'disk', 'square', 'diamond', 'line', 'octagon'
%       seSize     - ขนาดของ SE สำหรับ strel (ค่าเริ่มต้น: 5)
%                    สำหรับ 'disk'/'square'/'diamond' คือรัศมี/ความกว้าง
%
%   เอาต์พุต (struct — สะดวกผูกกับ App Designer):
%       morphResults.openedMask   - หลัง imopen  (ลบจุดขาวเล็กๆ)
%       morphResults.closedMask   - หลัง imclose (อุดช่องว่างเล็ก)
%       morphResults.erodedMask   - หลัง imerode
%       morphResults.dilatedMask  - หลัง imdilate
%       morphResults.filledMask   - หลัง imfill(..., 'holes') บน closed mask
%       morphResults.cleanedMask  - มาสก์มือสุดท้ายที่แนะนำให้ส่งต่อ Module 3+
%       morphResults.seShape      - รูปทรง SE ที่ใช้จริง
%       morphResults.seSize       - ขนาด SE ที่ใช้จริง
%
%   pipeline ที่เก็บใน cleanedMask:
%       open -> close -> fill holes  (เงามือเสถียรสำหรับดึง feature)
%
%   ตัวอย่าง (App Designer):
%       R = apply_morphology(app.SkinMask, 'disk', 5);
%       imshow(R.cleanedMask, 'Parent', app.MorphAxes);
%
%   Student 2 - Module 2: ปฏิบัติการทางสัณฐานวิทยา (Morphological Operations)

    %% -------------------- ค่าเริ่มต้น --------------------
    if nargin < 2 || isempty(seShape)
        seShape = 'disk';
    end
    if nargin < 3 || isempty(seSize)
        seSize = 5;
    end

    %% -------------------- ตรวจสอบอินพุต --------------------
    if nargin < 1 || isempty(binaryMask)
        error('apply_morphology:InvalidInput', ...
            'binaryMask is required and must not be empty.');
    end

    % รับได้ทั้งมาสก์ logical หรือตัวเลขจากขั้น segmentation ก่อนหน้า
    if islogical(binaryMask)
        bw = binaryMask;
    elseif isnumeric(binaryMask)
        bw = binaryMask ~= 0;
    else
        error('apply_morphology:InvalidType', ...
            'binaryMask must be logical or numeric.');
    end

    if ~ismatrix(bw)
        error('apply_morphology:InvalidSize', ...
            'binaryMask must be a 2-D image (HxW).');
    end

    if ~(ischar(seShape) || isstring(seShape))
        error('apply_morphology:InvalidSE', ...
            'seShape must be a character vector or string (e.g. ''disk'').');
    end
    seShape = char(seShape);

    if ~(isnumeric(seSize) && isscalar(seSize) && seSize > 0)
        error('apply_morphology:InvalidSESize', ...
            'seSize must be a positive scalar.');
    end

    %% -------------------- Structuring element --------------------
    % เทคโนโลยี: strel กำหนดรูปทรงละแวกที่ใช้ในปฏิบัติการสัณฐานวิทยา
    % (disk เป็นค่าเริ่มต้นที่ดีสำหรับมือที่ค่อนข้างมน)
    try
        se = strel(seShape, seSize);
    catch ME
        error('apply_morphology:StrelFailed', ...
            'Could not create structuring element strel(''%s'', %g): %s', ...
            seShape, seSize, ME.message);
    end

    %% -------------------- ปฏิบัติการสัณฐานวิทยาหลัก --------------------
    % Opening = กัดกร่อนแล้วขยาย → ลบจุดรบกวนสีขาวขนาดเล็ก
    openedMask = imopen(bw, se);

    % Closing = ขยายแล้วกัดกร่อน → อุดช่องว่างมืดขนาดเล็ก
    closedMask = imclose(bw, se);

    % Erosion หดพื้นหน้า (ช่วยแยกบริเวณที่เชื่อมกันหลวมๆ ได้)
    erodedMask = imerode(bw, se);

    % Dilation ขยายพื้นหน้า (ช่วยเชื่อมรอยขาดเล็กๆ บนมือ)
    dilatedMask = imdilate(bw, se);

    %% -------------------- อุดรู และมาสก์สุดท้ายที่ทำความสะอาดแล้ว --------------------
    % อุดรูภายใน closed mask (รูในฝ่ามือ/นิ้วที่พบบ่อย)
    filledMask = imfill(closedMask, 'holes');

    % มาสก์สุดท้ายสำหรับดึง feature / จำแนกขั้นถัดไป:
    % open (ลด noise) -> close (เชื่อมช่องว่าง) -> fill holes
    cleanedMask = imfill(imclose(openedMask, se), 'holes');

    %% -------------------- แพ็กผลลัพธ์เป็น struct สำหรับ GUI --------------------
    morphResults = struct();
    morphResults.openedMask  = openedMask;
    morphResults.closedMask  = closedMask;
    morphResults.erodedMask  = erodedMask;
    morphResults.dilatedMask = dilatedMask;
    morphResults.filledMask  = filledMask;
    morphResults.cleanedMask = cleanedMask;
    morphResults.seShape     = seShape;
    morphResults.seSize      = seSize;
end
