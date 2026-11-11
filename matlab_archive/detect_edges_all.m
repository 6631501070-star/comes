function edgeResults = detect_edges_all(grayOrMask)
%DETECT_EDGES_ALL รันตัวตรวจจับขอบแบบ Sobel, Prewitt และ Canny
%
%   เทคโนโลยี:
%     - Sobel / Prewitt: อิงเกรเดียนต์ด้วยเคอร์เนลขนาดเล็กเพื่อประมาณอนุพันธ์ของภาพ
%       เหมาะกับเส้นขอบมือที่ชัด
%     - Canny: หลายขั้นตอน (ทำให้เรียบ, หาเกรเดียนต์, non-maxima suppression,
%       hysteresis thresholding); มักได้เส้นขอบนิ้วที่สะอาดกว่า
%       สำหรับดึง feature ของ fingerspelling A-Z
%
%   รูปแบบการเรียกใช้:
%       edgeResults = detect_edges_all(grayOrMask)
%
%   อินพุต:
%       grayOrMask - ภาพเทา หรือมาสก์ไบนารี (HxW)
%                    รองรับ: uint8, uint16, single, double, logical
%                    ถ้าส่ง RGB (HxWx3) มา จะแปลงด้วย rgb2gray
%
%   เอาต์พุต (struct สำหรับ App Designer):
%       edgeResults.sobelEdges   - แผนที่ขอบแบบ logical จาก Sobel
%       edgeResults.prewittEdges - แผนที่ขอบแบบ logical จาก Prewitt
%       edgeResults.cannyEdges   - แผนที่ขอบแบบ logical จาก Canny
%
%   ตัวอย่าง (App Designer):
%       E = detect_edges_all(app.CleanedMask);
%       imshow(E.cannyEdges, 'Parent', app.EdgeAxes);
%
%   Student 2 - Module 2: การตรวจจับขอบ (Edge Detection)

    %% -------------------- ตรวจสอบอินพุต --------------------
    if nargin < 1 || isempty(grayOrMask)
        error('detect_edges_all:InvalidInput', ...
            'grayOrMask is required and must not be empty.');
    end

    if ~(isnumeric(grayOrMask) || islogical(grayOrMask))
        error('detect_edges_all:InvalidType', ...
            'grayOrMask must be numeric or logical.');
    end

    %% -------------------- แปลงเป็นภาพคล้ายเทา 2 มิติ --------------------
    img = grayOrMask;

    % RGB -> เทา เพราะ edge() ต้องการความเข้มแบบ 2 มิติ
    if ndims(img) == 3 && size(img, 3) == 3
        if isinteger(img)
            img = rgb2gray(img);
        else
            tmp = double(img);
            if max(tmp(:)) > 1
                tmp = tmp / 255;
            end
            tmp = min(max(tmp, 0), 1);
            img = rgb2gray(tmp);
        end
    elseif ~ismatrix(img)
        error('detect_edges_all:InvalidSize', ...
            'grayOrMask must be HxW or HxWx3.');
    end

    % มาสก์ logical -> double [0,1] ให้ตัวดำเนินการเกรเดียนต์เห็นการกระโดด 0/1
    if islogical(img)
        img = double(img);
    elseif isfloat(img)
        img = double(img);
        if max(img(:)) > 1
            img = img / 255;
        end
        img = min(max(img, 0), 1);
    elseif isinteger(img)
        % คงคลาสจำนวนเต็ม; edge() รองรับ uint8 และที่คล้ายกัน
    else
        error('detect_edges_all:UnsupportedType', ...
            'Unsupported image class: %s', class(img));
    end

    %% -------------------- ตัวตรวจจับขอบ --------------------
    % Sobel: เน้นเกรเดียนต์แนวทแยง/นอน/ตั้ง ด้วยเคอร์เนล 3x3
    sobelEdges = edge(img, 'Sobel');

    % Prewitt: คล้าย Sobel แต่ค่าน้ำหนักในเคอร์เนลต่างกันเล็กน้อย
    prewittEdges = edge(img, 'Prewitt');

    % Canny: โดยทั่วไปให้คุณภาพเส้นขอบนิ้วดีที่สุด
    cannyEdges = edge(img, 'Canny');

    %% -------------------- แพ็กผลลัพธ์เป็น struct สำหรับ GUI --------------------
    edgeResults = struct();
    edgeResults.sobelEdges   = sobelEdges;
    edgeResults.prewittEdges = prewittEdges;
    edgeResults.cannyEdges   = cannyEdges;
end
