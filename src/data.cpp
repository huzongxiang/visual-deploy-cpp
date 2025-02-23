#include "data.h"

// 定义颜色向量
std::vector<cv::Scalar> data_utils::colors;

size_t data_utils::vectorProduct(const std::vector<int64_t> &vector)
{
    if (vector.empty())
        return 0;

    size_t product = 1;
    for (const auto &element : vector)
        product *= element;

    return product;
}

std::wstring data_utils::charToWstring(const char *str)
{
    typedef std::codecvt_utf8<wchar_t> convert_type;//std::codecvt_utf8<wchar_t> 是一种 转换类型，用于将 UTF-8 字符串与 wchar_t 宽字符字符串之间进行相互转换。
    //在 Windows 系统中，wchar_t 通常是 UTF-16 编码。
    //在 Linux / Unix 系统中，wchar_t 通常是 UTF - 32 编码。
    std::wstring_convert<convert_type, wchar_t> converter;
    //std::wstring_convert 需要一个 编码转换类型（如 std::codecvt_utf8）和一个 宽字符类型（如 wchar_t）

    return converter.from_bytes(str);
}

void data_utils::loadNames(const std::vector<std::string>& classNames)
{
    // 初始化colors
    colors.clear();
    if (classNames.size() == 3) { // 交通灯用固定颜色
        colors = { cv::Scalar(0,255,0),    // Green
                   cv::Scalar(0,0,255),    // Red
                   cv::Scalar(0,255,255)}; // Yellow
    } else {
        // 否则使用随机颜色
        srand(time(0));
        for (size_t i = 0; i < classNames.size(); i++) {
            int b = rand() % 256;
            int g = rand() % 256;
            int r = rand() % 256;
            colors.push_back(cv::Scalar(b, g, r));
        }
    }
}

void data_utils::visualizeDetection(cv::Mat &im, std::vector<Detection> &results,
                               const std::vector<std::string> &classNames)
{
    cv::Mat image = im.clone();
    for (const Detection &result : results)
    {
        int x = result.box.x;
        int y = result.box.y;

        int conf = (int)std::round(result.confidence * 100);
        int classId = result.classId;
        std::string label = classNames[classId] + " 0." + std::to_string(conf);

        int baseline = 0;
        cv::Size size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 1.2, 1.5, &baseline);
        
        cv::rectangle(image, result.box, colors[classId], 2);
        cv::rectangle(image,
                      cv::Point(x, y), cv::Point(x + size.width, y + 30),
                      colors[classId], -1);
        cv::putText(image, label,
                    cv::Point(x, y - 3 + 25), cv::FONT_HERSHEY_SIMPLEX,
                    1.0, cv::Scalar(0, 0, 0), 3);
    }
    cv::addWeighted(im, 0.4, image, 0.6, 0, im);
}

void data_utils::letterbox(const cv::Mat &image, cv::Mat &outImage,
                      const cv::Size &newShape = cv::Size(640, 640),
                      const cv::Scalar &color = cv::Scalar(114, 114, 114),
                      bool auto_ = true,//是否根据步幅对填充尺寸进行自动调整
                      bool scaleFill = false,//是否强制将图像拉伸到目标尺寸（忽略长宽比）
                      bool scaleUp = true,//是否允许放大图像，如果为 false，图像只会缩小或保持原始尺寸
                      int stride = 32)//对齐步幅，用于控制填充的边缘尺寸
{
    cv::Size shape = image.size();
    //计算缩放比例
    float r = std::min((float)newShape.height / (float)shape.height,
                       (float)newShape.width / (float)shape.width);
    //如果 scaleUp 为 false，缩放比例 r 被限制为 1.0，确保图像不会被放大（仅会缩小或保持原尺寸
    if (!scaleUp)
        r = std::min(r, 1.0f);
    float ratio[2]{r, r};
    //调整图像尺寸
    int newUnpad[2]{(int)std::round((float)shape.width * r),
                    (int)std::round((float)shape.height * r)};
    //计算填充大小
    auto dw = (float)(newShape.width - newUnpad[0]);
    auto dh = (float)(newShape.height - newUnpad[1]);

    if (auto_)
    {
        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);
    }
    else if (scaleFill)
    {
        dw = 0.0f;
        dh = 0.0f;
        newUnpad[0] = newShape.width;
        newUnpad[1] = newShape.height;
        ratio[0] = (float)newShape.width / (float)shape.width;
        ratio[1] = (float)newShape.height / (float)shape.height;
    }

    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != newUnpad[0] && shape.height != newUnpad[1])
    {
        cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
    }

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    //添加填充
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

void data_utils::scaleCoords(cv::Rect &coords, const cv::Size &imageShape, const cv::Size &imageOriginalShape) {
    float gain = std::min((float)imageShape.height / (float)imageOriginalShape.height,
                          (float)imageShape.width / (float)imageOriginalShape.width);

    int pad[2] = {(int)(((float)imageShape.width - (float)imageOriginalShape.width * gain) / 2.0f),
                  (int)(((float)imageShape.height - (float)imageOriginalShape.height * gain) / 2.0f)};

    coords.x = (int)std::round(((float)(coords.x - pad[0]) / gain));
    coords.x = std::max(0, coords.x);
    coords.y = (int)std::round(((float)(coords.y - pad[1]) / gain));
    coords.y = std::max(0, coords.y);

    coords.width = (int)std::round(((float)coords.width / gain));
    coords.width = std::min(coords.width, imageOriginalShape.width - coords.x);
    coords.height = (int)std::round(((float)coords.height / gain));
    coords.height = std::min(coords.height, imageOriginalShape.height - coords.y);
}

template <typename T>
T data_utils::clip(const T &n, const T &lower, const T &upper)
{
    return std::max(lower, std::min(n, upper));
}

