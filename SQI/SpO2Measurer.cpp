// SpO2Measurer.cpp


/* Header */
//====================================================================================================

#include <Math/UniformNaturalCubicSpline.h>

#include <DSP/Filter/FilterFiltFilt.h>

#include <Vital/SpO2Measurer.h>

#include <DSP/Filter/Resampling.h>

//====================================================================================================


/* Member */
//====================================================================================================

namespace FHV
{
    class SpO2Measurer::Member
    {
        public: // Definition

            static const float ROI_WIDTH_RATIO;
            static const float ROI_HEIGHT_RATIO_UP;
            static const float ROI_HEIGHT_RATIO_DOWN;
            static const float IOU_THRESH;

            static const size_t SIZE_SEC      = 60;
            static const size_t SKIP_HEAD_SEC = 3;
            static const size_t SKIP_TAIL_SEC = 3;
            static const size_t START_SEC     = 30;

            typedef struct _wave_ {
                
                size_t start_id = 0;
                size_t end_id   = 0;
                size_t period   = 0;

                float norm_amp   = 0.0f;
                float noise_diff = 0.0f;
                float sn         = 0.0f;

            } _wave;

        public: // Function

            static bool updateRoI(const FaceDetector::_face_box& face_box, BBox::_box& roi_record, cv::Rect& roi);
            static bool extractData(const cv::Mat& image, std::vector<std::vector<float>>& signal);

            static cv::Rect calRoI(const FaceDetector::_face_box& face_box);

            static std::vector<_wave> extractWave(
                std::vector<float>::iterator& dc_begin, 
                std::vector<float>::iterator& dc_end,
                std::vector<float>::iterator& ac_begin, 
                std::vector<float>::iterator& origin_begin,
                std::vector<float>::iterator& sn_begin,
                size_t                        offset
            );

            static void removeUnpairedWave(std::vector<_wave>& src1, std::vector<_wave>& src2);

            static std::vector<size_t> getTroughIndices(const std::vector<float>& src);
            static void removePreiodicOutlier(std::vector<_wave>& wave);
            static void removeNonOverlapped(std::vector<_wave>& src, const std::vector<_wave>& dst);

            static _result estimate(_postproc& postproc, const std::vector<_wave>& w1, const std::vector<_wave>& w2);

            static bool setInitCurveRoR(const std::vector<_wave>& w1, const std::vector<_wave>& w2, BufferFIFO<float>& ror);
            static bool setInitCurveNoiseDiff(const std::vector<_wave>& w, BufferFIFO<float>& noise_diff);
            
            static BufferFIFO<float> getRoR(const std::vector<_wave>& w1, const std::vector<_wave>& w2);

            static BufferFIFO<float> smoothCurveRoR(const BufferFIFO<float>& ror, MovingAvg<float>& ror_ma);

            static float smoothRoR(float ror, float last_ror, const BufferFIFO<float>& ror_buffer, MovingAvg<float>& ror_ma);

            static float estimateSpO2(float ror);

            static float estimateSpO2(const BufferFIFO<float>& ror, MovingAvg<float>& spo2_ma);
    };

    const float SpO2Measurer::Member::ROI_WIDTH_RATIO       = 0.28f;
    const float SpO2Measurer::Member::ROI_HEIGHT_RATIO_UP   = 0.01f;
    const float SpO2Measurer::Member::ROI_HEIGHT_RATIO_DOWN = 0.14f;
    const float SpO2Measurer::Member::IOU_THRESH            = 0.95f;

    class SpO2Measurer::Filter
    {
        public: // Function

            Filter();

            bool filter(const std::vector<float>& src, std::vector<float>& dc, std::vector<float>& ac);

        private: // Function

            std::vector<float> getLowEnvelope(const std::vector<float>& src);

        private: // Variable

            std::shared_ptr<FilterFiltFilt> bpf_1_1, bpf_1_2, bpf_2_1, bpf_2_2;
            std::shared_ptr<FilterFiltFilt> lpf_1, lpf_2;
    };
}

//====================================================================================================


/* Public Function */
//====================================================================================================

namespace FHV
{
    //------------------------------------------------------------------------------------------------

    SpO2Measurer::SpO2Measurer(size_t fps, size_t window_length)
    {
        size_t max_size;

        this->fps               = fps;
        this->window_length     = window_length;
        this->sampling_interval = (float)TARGET_FPS / (float)this->fps;
        this->sampling_num      = this->fps + 1;

        max_size = Member::SIZE_SEC * TARGET_FPS;

        this->long_in[0].setMaxSize(max_size);
        this->long_in[1].setMaxSize(max_size);
        this->long_in[2].setMaxSize(max_size);
        this->sn_in.setMaxSize(max_size);

        this->sample_in[0].setMaxSize(this->sampling_num);
        this->sample_in[1].setMaxSize(this->sampling_num);
        this->sample_in[2].setMaxSize(this->sampling_num);

        this->filter = std::make_unique<Filter>();

        this->reset();
    }

    //------------------------------------------------------------------------------------------------

    SpO2Measurer::SpO2Measurer(const SpO2Measurer& another) 
    {

    }

    //------------------------------------------------------------------------------------------------

    SpO2Measurer::~SpO2Measurer()
    {

    }

    //------------------------------------------------------------------------------------------------

    size_t SpO2Measurer::getFPS()
    {
        return this->fps;
    }

    //------------------------------------------------------------------------------------------------

    void SpO2Measurer::resetFPS(size_t fps)
    {
        this->fps = fps;

        this->sampling_interval = (float)TARGET_FPS / (float)this->fps;
        this->sampling_num = this->fps + 1;

        this->sample_in[0].setMaxSize(this->sampling_num);
        this->sample_in[1].setMaxSize(this->sampling_num);
        this->sample_in[2].setMaxSize(this->sampling_num);

        this->reset();
    }

    //------------------------------------------------------------------------------------------------

    void SpO2Measurer::reset()
    {
        this->signal.clear();
        this->signal.resize(3);
        this->signal[0].assign(this->window_length, 0.0f);
        this->signal[1].assign(this->window_length, 0.0f);
        this->signal[2].assign(this->window_length, 0.0f);

        this->long_in[0].clear();
        this->long_in[1].clear();
        this->long_in[2].clear();
        this->sn_in.clear();

        this->sample_in[0].clear();
        this->sample_in[1].clear();
        this->sample_in[2].clear();

        this->idle_count = 0;
        this->sn         = 0;

        this->postproc.start   = false;
        this->postproc.last_sn = 0.0f;
        this->postproc.noise_diff.clear();
        this->postproc.ror.clear();

        this->postproc.curr_result.reset();
        this->postproc.last_result.reset();
    }

    //------------------------------------------------------------------------------------------------

    bool SpO2Measurer::invoke(SpO2Measurer& measurer, const cv::Mat& image, const FaceDetector::_face_box& face_box)
    {
        bool ret = false;

        size_t i, min_len;
        size_t head_offset, tail_offset;

        cv::Rect roi;
        cv::Mat roi_image;

        std::unique_ptr<UniformNaturalCubicSpline<float>> spline[3];

        std::vector<float> long_in[3];
        std::vector<float> dc[3], ac[3];
        std::vector<float> sn_in;

        std::vector<Member::_wave> wave_r, wave_g;

        std::vector<float>::iterator dc_begin, dc_end, ac_begin, origin_begin, sn_begin;

        if (Member::updateRoI(face_box, measurer.roi_record, roi))
        {
            ret = true;

            roi &= cv::Rect(0, 0, image.cols, image.rows);

            if (!roi.empty())
            {
                roi_image = image(roi).clone();
                Member::extractData(roi_image, measurer.signal);
                
                if (measurer.fps != TARGET_FPS)
                {                   
                    measurer.sample_in[0].push(measurer.signal[0].back());
                    measurer.sample_in[1].push(measurer.signal[1].back());
                    measurer.sample_in[2].push(measurer.signal[2].back());

                    if (measurer.sample_in[2].size() >= measurer.sampling_num)
                    {
                        spline[0] = std::make_unique<UniformNaturalCubicSpline<float>>(measurer.sampling_num, measurer.sample_in[0].cloneToVector(), 0.0f, measurer.sampling_interval);
                        spline[1] = std::make_unique<UniformNaturalCubicSpline<float>>(measurer.sampling_num, measurer.sample_in[1].cloneToVector(), 0.0f, measurer.sampling_interval);
                        spline[2] = std::make_unique<UniformNaturalCubicSpline<float>>(measurer.sampling_num, measurer.sample_in[2].cloneToVector(), 0.0f, measurer.sampling_interval);

                        for (i = 0; i < TARGET_FPS; i++)
                        {
                            measurer.long_in[0].push(spline[0]->at((float)i));
                            measurer.long_in[1].push(spline[1]->at((float)i));
                            measurer.long_in[2].push(spline[2]->at((float)i));
                            measurer.sn_in.push((float)measurer.sn++);
                            measurer.idle_count++;
                        }

                        measurer.sample_in[0].clear();
                        measurer.sample_in[1].clear();
                        measurer.sample_in[2].clear();
                        measurer.sample_in[0].push(measurer.signal[0].back());
                        measurer.sample_in[1].push(measurer.signal[1].back());
                        measurer.sample_in[2].push(measurer.signal[2].back());
                    }
                    
                }
                else
                {
                    measurer.long_in[0].push(measurer.signal[0].back());
                    measurer.long_in[1].push(measurer.signal[1].back());
                    measurer.long_in[2].push(measurer.signal[2].back());
                    measurer.sn_in.push((float)measurer.sn++);
                    measurer.idle_count++;
                }
                
                min_len = Member::START_SEC * TARGET_FPS;

                if (measurer.idle_count >= TARGET_FPS)
                {
                    measurer.idle_count = 0;

                    if (measurer.long_in[2].size() >= min_len)
                    {
                        long_in[0] = measurer.long_in[0].cloneToVector();
                        long_in[1] = measurer.long_in[1].cloneToVector();
                        long_in[2] = measurer.long_in[2].cloneToVector();
                        sn_in      = measurer.sn_in.cloneToVector();

                        measurer.filter->filter(long_in[0], dc[0], ac[0]);
                        measurer.filter->filter(long_in[1], dc[1], ac[1]);
                        measurer.filter->filter(long_in[2], dc[2], ac[2]);

                        head_offset = Member::SKIP_HEAD_SEC * TARGET_FPS;
                        tail_offset = Member::SKIP_TAIL_SEC * TARGET_FPS;

                        dc_begin     = dc[2].begin() + head_offset;
                        dc_end       = dc[2].end() - tail_offset;
                        ac_begin     = ac[2].begin() + head_offset;
                        origin_begin = long_in[2].begin() + head_offset; 
                        sn_begin     = sn_in.begin() + head_offset;

                        wave_r = Member::extractWave(dc_begin, dc_end, ac_begin, origin_begin, sn_begin, head_offset);

                        dc_begin     = dc[1].begin() + head_offset;
                        dc_end       = dc[1].end() - tail_offset;
                        ac_begin     = ac[1].begin() + head_offset;
                        origin_begin = long_in[1].begin() + head_offset; 
                        sn_begin     = sn_in.begin() + head_offset;

                        wave_g = Member::extractWave(dc_begin, dc_end, ac_begin, origin_begin, sn_begin, head_offset);

                        Member::removeUnpairedWave(wave_r, wave_g);

                        measurer.postproc.curr_result = Member::estimate(measurer.postproc, wave_g, wave_r);
                    }
                }
            }
        }

        return ret;
    }

    //------------------------------------------------------------------------------------------------

    bool SpO2Measurer::invoke(const cv::Mat& image, const FaceDetector::_face_box& face_box)
    {
        return invoke(*this, image, face_box);
    }

    //------------------------------------------------------------------------------------------------

    float SpO2Measurer::getSpO2()
    {
        return this->postproc.curr_result.spo2;
    }

    //------------------------------------------------------------------------------------------------

    float SpO2Measurer::getScore()
    {
        return this->postproc.curr_result.score;
    }

    //------------------------------------------------------------------------------------------------

}

//====================================================================================================


/* Private Function */
//====================================================================================================

namespace FHV
{
    //------------------------------------------------------------------------------------------------

    bool SpO2Measurer::Member::updateRoI(const FaceDetector::_face_box& face_box, BBox::_box& roi_record, cv::Rect& roi)
    {
        bool ret = false;

        if ((face_box.landmark_num > 0) && (face_box.landmark_num <= MTCNN::LANDMARK_NUM))
        {
            ret = true;

            if (BBox::IoU(face_box.bbox.box, roi_record) < IOU_THRESH)
            {
                roi = calRoI(face_box);
                roi_record = BBox::RectToBox(roi);
            }
            else
            {
                roi = BBox::BoxToRect(roi_record);
            }
        }

        return ret;
    }

    //------------------------------------------------------------------------------------------------

    bool SpO2Measurer::Member::extractData(const cv::Mat& image, std::vector<std::vector<float>>& signal)
    {
        bool ret = false;

        size_t i;
        cv::Scalar sum;

        if (!image.empty())
        {
            if (image.channels() == signal.size())
            {
                ret = true;
                sum = cv::sum(image);

                for (i = 0; i < signal.size(); i++)
                {
                    memmove(&signal[i][0], &signal[i][1], sizeof(float) * (signal[i].size() - 1));
                    signal[i].back() = (float)(sum[i] / (double)(image.cols * image.rows));
                }
            }            
        }

        return ret;
    }

    //------------------------------------------------------------------------------------------------

    cv::Rect SpO2Measurer::Member::calRoI(const FaceDetector::_face_box& face_box)
    {
        cv::Rect roi;
        
        const cv::Point2f* nose;
        
        float w, h;

        nose = &face_box.landmarks[MTCNN::NOSE];

        w = BBox::getWidth(face_box.bbox.box);
        h = BBox::getHeight(face_box.bbox.box);

        roi.x = (int)(nose->x - w * ROI_WIDTH_RATIO);
        roi.y = (int)(nose->y - h * ROI_HEIGHT_RATIO_UP);

        roi.width  = (int)(nose->x + w * ROI_WIDTH_RATIO)       - roi.x;
        roi.height = (int)(nose->y + h * ROI_HEIGHT_RATIO_DOWN) - roi.y;

        return roi;
    }

    //------------------------------------------------------------------------------------------------

    std::vector<SpO2Measurer::Member::_wave> SpO2Measurer::Member::extractWave(
        std::vector<float>::iterator& dc_begin,
        std::vector<float>::iterator& dc_end,
        std::vector<float>::iterator& ac_begin,
        std::vector<float>::iterator& origin_begin,
        std::vector<float>::iterator& sn_begin,
        size_t                        offset
    )
    {
        size_t i, j;

        float len;

        std::vector<float>::iterator ac_end, origin_end, sn_end;
        std::vector<float>::iterator ac_max_it;

        std::vector<float>  src_norm;
        std::vector<size_t> trough_indices;

        _wave tmp;
        std::vector<_wave> wave;

        len        = std::distance(dc_begin, dc_end);
        ac_end     = ac_begin + len;
        origin_end = origin_begin + len;
        sn_end     = sn_begin + len;

        src_norm.resize(len);
        std::transform(ac_begin, ac_end, dc_begin, src_norm.begin(), std::divides<float>());

        trough_indices = getTroughIndices(src_norm);
        wave.reserve(trough_indices.size());

        for (i = 1; i < trough_indices.size(); i++)
        {
            tmp.start_id = trough_indices[i - 1];
            tmp.end_id   = trough_indices[i];
            tmp.period   = tmp.end_id - tmp.start_id;

            ac_max_it = std::max_element(ac_begin + tmp.start_id, ac_begin + tmp.end_id);

            tmp.norm_amp = *ac_max_it / *(dc_begin + (size_t)std::distance(ac_begin, ac_max_it));

            tmp.noise_diff = 0.0f;

            for (j = tmp.start_id; j <= tmp.end_id; j++) 
            {
                tmp.noise_diff += std::abs(*(origin_begin + j) - *(ac_begin + j) - *(dc_begin + j));
            }

            tmp.noise_diff /= tmp.period;
            tmp.sn = *(sn_begin + tmp.start_id);

            tmp.start_id += offset;
            tmp.end_id   += offset;
            wave.push_back(tmp);
        }

        removePreiodicOutlier(wave);

        return wave;
    }

    //------------------------------------------------------------------------------------------------

    void SpO2Measurer::Member::removeUnpairedWave(std::vector<_wave>& src1, std::vector<_wave>& src2)
    {
        removeNonOverlapped(src1, src2);
        removeNonOverlapped(src2, src1);
    }

    //------------------------------------------------------------------------------------------------

    std::vector<size_t> SpO2Measurer::Member::getTroughIndices(const std::vector<float>& src)
    {
        size_t i;

        std::vector<size_t> indices;

        for (i = 1; i < (src.size() - 1); i++)
        {
            if ((src[i] < src[i - 1]) && (src[i] < src[i + 1])) 
            {
                indices.push_back(i);
            }
        }

        return indices;
    }

    //------------------------------------------------------------------------------------------------

    void SpO2Measurer::Member::removePreiodicOutlier(std::vector<_wave>& wave)
    {
        size_t duration_total, duration_avg;
        size_t lower, upper;

        std::vector<_wave>::iterator it;

        if (!wave.empty())
        {
            duration_total = wave.back().end_id - wave.front().start_id + 1;
            duration_avg   = duration_total / wave.size();

            lower = (size_t)std::floor((float)duration_total / (float)(wave.size() + 5));
            upper = (size_t)std::ceil((float)duration_total / (float)(wave.size() - 5));

            it = wave.begin();

            while (it != wave.end())
            {
                if ((it->period >= lower) && (it->period <= upper))
                {
                    it++;
                }
                else
                {
                    it = wave.erase(it);
                }
            }
        }
    }

    //------------------------------------------------------------------------------------------------

    void SpO2Measurer::Member::removeNonOverlapped(std::vector<_wave>& src, const std::vector<_wave>& dst)
    {
        bool is_overlapped;

        std::vector<_wave>::iterator it1;
        std::vector<_wave>::const_iterator it2;

        size_t id1, id2;
        size_t overlap_len1, overlap_len2;
        float overlap_ratio1, overlap_ratio2;

        it1 = src.begin();

        while (it1 != src.end())
        {
            is_overlapped = false;

            for (it2 = dst.begin(); it2 != dst.end(); it2++) 
            {
                if (it2->start_id > it1->end_id) break;

                id1 = MAX(it1->start_id, it2->start_id);
                id2 = MIN(it1->end_id, it2->end_id);

                if (id1 < id2)
                {
                    overlap_len1 = MIN(id2, it1->end_id) - MAX(id1, it1->start_id);
                    overlap_len2 = MIN(id2, it2->end_id) - MAX(id1, it2->start_id);
                    overlap_ratio1 = (float)overlap_len1 / (float)it1->period;
                    overlap_ratio2 = (float)overlap_len2 / (float)it2->period;

                    if ((overlap_ratio1 > 0.7f) && (overlap_ratio2 > 0.7f))
                    {
                        is_overlapped = true;
                        break;
                    }
                }
            }

            if (!is_overlapped) 
            {
                it1 = src.erase(it1);
            }
            else 
            {
                it1++;
            }
        }
    }

    //------------------------------------------------------------------------------------------------

    SpO2Measurer::_result SpO2Measurer::Member::estimate(_postproc& postproc, const std::vector<_wave>& w1, const std::vector<_wave>& w2)
    {
        size_t i;

        BufferFIFO<float> smooth_ror_curve;

        _result result;

        float ror;

        if (!postproc.start)
        {
            if (setInitCurveRoR(w1, w2, postproc.ror))
            {
                postproc.start = true;
                setInitCurveNoiseDiff(w2, postproc.noise_diff);

                smooth_ror_curve = smoothCurveRoR(postproc.ror, postproc.ror_ma);

                result.estimated = true;
                result.ror       = smooth_ror_curve.cloneToVector().back();
                result.spo2      = estimateSpO2(smooth_ror_curve, postproc.spo2_ma);
                result.score     = MAX(1.0f - postproc.noise_diff.calMean(), 0.0f);

                postproc.score_ma.reset(4);
                postproc.last_result = result;
                postproc.last_sn     = w1.back().sn;
            }
        }
        else
        {
            result = postproc.last_result;

            for (i = 0; (i < w1.size()) && (i < w2.size()); i++)
            {
                if (w1[i].sn >= postproc.last_sn)
                {
                    ror = w1[i].norm_amp / (w2[i].norm_amp + FLT_EPSILON);

                    if (!std::isnan(ror))
                    {
                        postproc.ror.push(ror);
                        postproc.noise_diff.push(w2[i].noise_diff);
                        result.estimated = true;
                        result.ror   = smoothRoR(ror, postproc.last_result.ror, postproc.ror, postproc.ror_ma);
                        result.spo2  = postproc.spo2_ma.update(estimateSpO2(result.ror));
                        result.score = postproc.score_ma.update(MAX(1.0f - postproc.noise_diff.calMean(), 0.0f));
                        postproc.last_result = result;
                    }

                    postproc.last_sn = w1[i].sn;
                }
            }
        }

        return result;
    }

    //------------------------------------------------------------------------------------------------

    bool SpO2Measurer::Member::setInitCurveRoR(const std::vector<_wave>& w1, const std::vector<_wave>& w2, BufferFIFO<float>& ror)
    {
        bool ret = false;

        size_t i;

        std::vector<float> ror_curve;

        ror_curve.resize(w1.size());

        for (i = 0; i < ror_curve.size(); i++)
        {
            ror_curve[i] = w1[i].norm_amp / (w2[i].norm_amp + FLT_EPSILON);
        }

        ror_curve.erase(std::remove_if(ror_curve.begin(), ror_curve.end(), [](double v) { return std::isnan(v); }), ror_curve.end());
        
        if (!ror_curve.empty()) 
        {
            ret = true;
            ror = ror_curve;         
        }

        return ret;
    }

    //------------------------------------------------------------------------------------------------

    bool SpO2Measurer::Member::setInitCurveNoiseDiff(const std::vector<_wave>& w, BufferFIFO<float>& noise_diff)
    {
        bool ret = !w.empty();

        size_t i;

        noise_diff.setMaxSize(w.size());

        for (i = 0; i < w.size(); i++)
        {
            noise_diff.push(w[i].noise_diff);
        }

        return ret;
    }

    //------------------------------------------------------------------------------------------------

    BufferFIFO<float> SpO2Measurer::Member::getRoR(const std::vector<_wave>& w1, const std::vector<_wave>& w2)
    {
        size_t i;
        std::vector<float> ror_curve;

        ror_curve.resize(w1.size());

        for (i = 0; i < ror_curve.size(); i++)
        {
            ror_curve[i] = w1[i].norm_amp / w2[i].norm_amp;
        }

        ror_curve.erase(std::remove_if(ror_curve.begin(), ror_curve.end(), [](double v) { return std::isnan(v); }), ror_curve.end());

        return ror_curve;
    }

    //------------------------------------------------------------------------------------------------

    BufferFIFO<float> SpO2Measurer::Member::smoothCurveRoR(const BufferFIFO<float>& ror, MovingAvg<float>& ror_ma)
    {
        std::vector<float> ret;

        size_t i;

        float mean, std;
        float upper, lower;
        float new_ror, v;

        ret.reserve(ror.size());

        mean = ror.calMean();
        std  = ror.calStd();
        
        upper = mean + std;
        lower = mean - std;

        ror_ma.reset(15, mean);

        new_ror = mean;

        for (i = 0; i < ror.size(); i++)
        {
            v = ror[i];
            new_ror = ror_ma.update(((v < upper) && (v > lower)) ? v : new_ror);
            ret.push_back(new_ror);
        }

        return ret;
    }

    //------------------------------------------------------------------------------------------------

    float SpO2Measurer::Member::smoothRoR(float ror, float last_ror, const BufferFIFO<float>& ror_buffer, MovingAvg<float>& ror_ma)
    {
        float mean  = ror_buffer.calMean();
        float std   = ror_buffer.calStd();
        float upper = mean + std;
        float lower = mean - std;
        return ror_ma.update(((ror > lower) && (ror < upper)) ? ror : last_ror);
    }

    //------------------------------------------------------------------------------------------------

    float SpO2Measurer::Member::estimateSpO2(float ror)
    {
        return (ror < 2.0f) ? MAX(std::round(67.334f + 13.333f * ror), 90.0f) : MIN(std::round(74.0f + 10.0f * ror), 99.0f);
    }

    //------------------------------------------------------------------------------------------------

    float SpO2Measurer::Member::estimateSpO2(const BufferFIFO<float>& ror, MovingAvg<float>& spo2_ma)
    {
        float spo2 = -1.0f;

        size_t i;

        std::vector<float> spo2_curve;

        if (ror.size())
        {
            spo2_curve.reserve(ror.size());

            for (i = 0; i < ror.size(); i++)
            {
                spo2_curve.push_back(estimateSpO2(ror[i]));
            }

            spo2_ma.reset(4, spo2_curve.front());

            for (i = 0; i < spo2_curve.size(); i++)
            {
                spo2 = spo2_ma.update(spo2_curve[i]);
            }
        }

        return spo2;
    }

    //------------------------------------------------------------------------------------------------

    SpO2Measurer::Filter::Filter()
    {
        float b[FilterFiltFilt::ORDER], a[FilterFiltFilt::ORDER];

        b[0] = 0.0845285022549549f; b[1] = -0.121950128357991f; b[2] = 0.0845285022549182f;
        a[0] = 1.0000000000000000f; a[1] = -1.601144146472060f; a[2] = 0.7105490363641240f;
        this->bpf_1_1 = std::make_shared<FilterFiltFilt>(b, a);

        b[0] = 1.0f; b[1] = -1.99398845353662f; b[2] = 1.000000000032800f;
        a[0] = 1.0f; a[1] = -1.82138146138240f; a[2] = 0.848447158367502f;
        this->bpf_1_2 = std::make_shared<FilterFiltFilt>(b, a);

        b[0] = 1.0f; b[1] = -1.81347339575825f; b[2] = 1.000000000004120f;
        a[0] = 1.0f; a[1] = -1.78973678436716f; a[2] = 0.925518542090677f;
        this->bpf_2_1 = std::make_shared<FilterFiltFilt>(b, a);

        b[0] = 1.0f; b[1] = -1.98017368340265f; b[2] = 0.999999999963516f;
        a[0] = 1.0f; a[1] = -1.94050007708460f; a[2] = 0.966546972025427f;
        this->bpf_2_2 = std::make_shared<FilterFiltFilt>(b, a);

        b[0] = 0.0883470268178469f; b[1] = -0.164143046303472f; b[2] = 0.0883470268178472f;
        a[0] = 1.0000000000000000f; a[1] = -1.739092711879470f; a[2] = 0.7621454218853940f;
        this->lpf_1 = std::make_shared<FilterFiltFilt>(b, a);

        b[0] = 1.0f; b[1] = -1.97488662411634f; b[2] = 0.999999999999997f;
        a[0] = 1.0f; a[1] = -1.92789173207993f; a[2] = 0.941564665317382f;
        this->lpf_2 = std::make_shared<FilterFiltFilt>(b, a);
    }

    //------------------------------------------------------------------------------------------------

    bool SpO2Measurer::Filter::filter(const std::vector<float>& src, std::vector<float>& dc, std::vector<float>& ac)
    {
        bool ret = false;

        std::vector<float> envelope;

        if (src.size() >= 512)
        {
            ret = true;

            dc.clear();
            dc.resize(src.size());

            this->bpf_1_1->filter(src, ac);
            this->bpf_1_2->filter(ac, ac);
            this->bpf_2_1->filter(ac, ac);
            this->bpf_2_2->filter(ac, ac);

            envelope = this->getLowEnvelope(ac);

            std::transform(ac.begin(), ac.end(), envelope.begin(), ac.begin(), std::minus<float>());
            std::transform(src.begin(), src.end(), ac.begin(), dc.begin(), std::minus<float>());

            lpf_1->filter(dc, dc);
            lpf_2->filter(dc, dc);
        }

        return ret;
    }

    //------------------------------------------------------------------------------------------------

    std::vector<float> SpO2Measurer::Filter::getLowEnvelope(const std::vector<float>& src)
    {
        size_t i, j;

        std::unique_ptr<UniformNaturalCubicSpline<float>> spline;

        int len;
        std::vector<int> indices;
        std::vector<float> ys, envelope;
        std::vector<float>::iterator it;

        float v;
        
        envelope.assign(src.size(), 0.0f);

        for (i = 1; i < (src.size() - 1); i++) 
        {
            if ((src[i] < src[i - 1]) && (src[i] < src[i + 1]))
            {
                indices.push_back(i);
            }
        }

        if (!indices.empty()) 
        {
            std::fill(envelope.begin(), envelope.begin() + indices.front(), src[indices.front()]);

            for (i = 0; i < (indices.size() - 1); i++) 
            {
                v = 0.5f * (src[indices[i]] + src[indices[i + 1]]);

                ys.resize(2);
                ys[0] = src[indices[i]];
                ys[1] = src[indices[i + 1]];

                len = indices[i + 1] - indices[i];

                spline = std::make_unique<UniformNaturalCubicSpline<float>>(2, ys, 0.0f, (float)len);

                j = 0;

                for (it = envelope.begin() + indices[i]; it != envelope.begin() + indices[i + 1]; it++) 
                {
                    *it = spline->at((float)j);
                    j++;
                }
            }

            std::fill(envelope.begin() + indices.back(), envelope.end(), src[indices.back()]);
        }

        return envelope;
    }

    //------------------------------------------------------------------------------------------------
}

//====================================================================================================