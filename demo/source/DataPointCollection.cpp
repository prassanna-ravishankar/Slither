#include "DataPointCollection.h"

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
  std::istream & getline_(std::istream & in, std::string & out)
  {
    out.clear();
    char c;
    while(in.get(c).good())
    {
      if(c == '\n')
      {
        c = in.peek();
        if(in.good())
        {
          if(c == '\r')
          {
            in.ignore();
          }
        }
        break;
      }
      out.append(1,c);
    }
    return in;
  }
            /*

  std::auto_ptr<DataPointCollection> DataPointCollection::Load(std::istream& r, int dataDimension, DataDescriptor::e descriptor)
  {
    bool bHasTargetValues = (descriptor & DataDescriptor::HasTargetValues) == DataDescriptor::HasTargetValues;
    bool bHasClassLabels = (descriptor & DataDescriptor::HasClassLabels) == DataDescriptor::HasClassLabels;

    std::auto_ptr<DataPointCollection> result = std::auto_ptr<DataPointCollection>(new DataPointCollection());
    result->dimension_ = dataDimension;


    unsigned int elementsPerLine = (bHasClassLabels ? 1 : 0) + dataDimension + (bHasTargetValues ? 1 : 0);

    std::string line;
    while (r.good())
    {
      getline_(r, line);

      if(r.fail())
        throw std::runtime_error("Failed to read line.");

      std::vector<std::string> elements;
      tokenize(line, elements, "\t"); // split tab-delimited line

      if (elements.size() != elementsPerLine)
        throw std::runtime_error("Encountered line with unexpected number of elements.");

      int index = 0;

      if (bHasClassLabels)
      {
        if(elements[index]!="")
        {
          if (result->labelIndices_.find(elements[index])==result->labelIndices_.end())
            result->labelIndices_.insert(std::pair<std::string, int>(elements[index], result->labelIndices_.size()));

          result->labels_.push_back(result->labelIndices_[elements[index++]]);
          result->uniqueClasses_.insert(result->labelIndices_[elements[index]]); //TODO : CHECK in c++
        }
        else
        {
          // cast necessary in g++ because std::vector<int>::push_back() takes a reference
          result->labels_.push_back((int)(DataPointCollection::UnknownClassLabel));
          result->uniqueClasses_.insert((int)(DataPointCollection::UnknownClassLabel)); //TODO:Check in c++
          index++;
        }
      }

      cv::Mat rowMat = cv::Mat(1,dataDimension,CV_32FC1);

      for (int i = 0; i < dataDimension; i++)
      {

        float x = to_float(elements[index++]);
        rowMat.at<float>(0,i) =x;
//        result->data_.push_back(x);
      }

      if (bHasTargetValues)
      {
        float t = to_float(elements[index++]);
        result->targets_.push_back(t);
      }

      result->dataMat.push_back(rowMat);
    }

    for(int i = 0;i<result->labels_.size();i++)
      std::cout<<result->labels_[i]<<" ";
    std::cout<<std::endl<<result->labels_.size();


    return result;
  }*/

  std::auto_ptr<DataPointCollection> DataPointCollection::Load(const std::string &filename)
  {
    std::auto_ptr<DataPointCollection> result = std::auto_ptr<DataPointCollection>(new DataPointCollection());

    cv::Ptr<cv::ml::TrainData> all_data;

    const cv::String bla = cv::String();
    try
    {
      all_data = cv::ml::TrainData::loadFromCSV(filename,0,0,-1,bla,'\t','?');
    }
    catch (cv::Exception & e)
    {
      std::cout<<"An Exception while reading has occurred: "<<e.what()<<std::endl;
    }


    cv::Mat responses = all_data->getResponses().t();

    const float* p = responses.ptr<float>(0);
    result->labels_ = std::vector<int> (p, p + responses.cols);


    //HACK - REmove it before it messes up the data
    for(int i=0;i<result->labels_.size();i++)
      result->labels_[i] = result->labels_[i]>0?1:0;


    result->uniqueClasses_ = std::set<int> (result->labels_.begin(), result->labels_.end());



    result->dataMat = cv::Mat(all_data->getSamples());
    result->dimension_ = all_data->getNVars();
    result->targets_.clear();


    all_data.release();


    return result;




  }



  std::auto_ptr<DataPointCollection> DataPointCollection::LoadPatches(const std::string &filename, const std::string &img_folder, const std::string &ann_folder)
  {
    std::auto_ptr<DataPointCollection> result = std::auto_ptr<DataPointCollection>(new DataPointCollection());

    cv::Ptr<cv::ml::TrainData> all_data;



    std::ifstream fin (filename.c_str());
    std::string delimiter = ":";
    std::string folder_name_images = filename.substr(0,filename.rfind("/")+1)+"Images/";

    std::cout<<"FOLDER is : "<<folder_name_images<<std::endl;


    //Init NNParams
    //Change to init it parameter-wise
    std::string model_file="/home/prassanna/Libraries/caffe-master/models/vggnet/VGG_CNN_F_copy.prototxt";
    std::string mean_file="";
    std::string trained_file="/home/prassanna/Libraries/caffe-master/models/vggnet/VGG_CNN_F.caffemodel";
    std::string label_file="/home/prassanna/Libraries/caffe-master/data/ilsvrc12/synset_words.txt";
    Classifier classifier(model_file, trained_file, mean_file, label_file);

    result->total_size = 0;

    int count_data=0;

    //cv::namedWindow("Labels");

    while(!fin.eof())
    {

      std::string labels;
      std::string line;
      std::getline(fin, line);
      int pos = 0;
      int lbl = 0;

      std::string image_name = img_folder + line;
      std::string ann_name = ann_folder + line;

      result-> filenames.push_back(line);


      if(!labels.empty())
        lbl = std::stoi(labels);


      if(!line.empty()) {

        //cv::ImreadModes ::IMREAD_ANYCOLOR

        cv::Mat img = cv::imread(image_name);
        cv::Mat ann_gt = cv::imread(ann_name,CV_LOAD_IMAGE_GRAYSCALE);

        if(img.empty() || ann_gt.empty())
          continue;

        //For Kitti only
        cv::Mat ann  = cv::Mat(ann_gt.rows, ann_gt.cols, CV_8U);
        ann = (ann_gt>=1)-254;
        std::vector<cv::Mat> image_feats = classifier.forwardPassVector(img,2); //is CV_32F
        cv::Mat hypercolumn = cv::Mat(1, image_feats.size(), CV_32FC1);

        //TODO :: Calculate slic segments


        /*
        cv::Mat label_contours;
        slic->getLabelContourMask(label_contours);
        cv::imshow("Labels", label_contours);
        cv::waitKey();

         */



        //Going through the superpixels, calculating superpixel hypercolumns



        //TODO :: Correspond pixel to image to segment
        // Introduce image index vector
        // Introduce Superpixel index vector

        int previous_superpixels=0;
        if(!result->superpixelFeats_1.empty())
          previous_superpixels = result->superpixelFeats_1.rows;

        cv::Mat ann_small, img_small;
        //cv::resize(channel, out, img.size());
        cv::resize(ann, ann_small, image_feats[0].size());
        cv::resize(img, img_small, image_feats[0].size());



        //SUPERPIXEL PART
        cv::Mat labels;
        int n = 800;
        float region_size = (int)std::sqrt((img.size().height*img.size().width) / n);
        cv::Ptr<cv::ximgproc::SuperpixelSLIC>slic = cv::ximgproc::createSuperpixelSLIC(img_small, cv::ximgproc::SLIC, region_size,10.0f);
        slic->iterate();
        slic->enforceLabelConnectivity();
        slic->getLabels(labels);
        double minVal, maxVal;
        cv::minMaxIdx(labels,&minVal, &maxVal);
        cv::Mat superpixel_descriptors = cv::Mat::zeros(maxVal+1, image_feats.size(), CV_32FC1);
        std::vector<int> superpixel_sizes;
        superpixel_sizes.resize(maxVal+1, 0);

        for(int y=0;y<image_feats[0].rows;y++)
            for(int x=0;x<image_feats[0].cols;x++)
            {
              int lbl = (int) ann_small.at<uchar>(y,x);
                for(int c=0;c<image_feats.size();c++)
                  hypercolumn.at<float>(c) = image_feats[c].at<float>(y,x);

              int superpixel_label = labels.at<int>(y,x);
              result->dataMat.push_back(hypercolumn);
              result->labels_.push_back(lbl);
              result->superpixel_pixel_map.push_back(previous_superpixels+superpixel_label);

              //result->image_idx_pixel.push_back(count_data);


              //Adding superpixel label
              superpixel_descriptors.row(superpixel_label) = superpixel_descriptors.row(superpixel_label)+ hypercolumn;
              superpixel_sizes[superpixel_label]++;

            }

        result->pixel_response_rows.push_back(result->dataMat.rows);


        //Average Superpixels
        for(int t=0;t<superpixel_sizes.size();t++)
        {
          result->superpixelFeats_1.push_back( (superpixel_descriptors.row(t) / superpixel_sizes[t]) );
          //result->superpixel_pixel_map.push_back(result->superpixelFeats_1.rows-1);
        }



        //result->patches.push_back(img);
        result->image_rows.push_back(img_small.rows);
        result->image_cols.push_back(img_small.cols);

        /*
        //To Debug
        int wa = result->dataMat.rows;
        int wb = result->labels_.size();
        int wc = result->image_idx_pixel.size();
        int wd = result->superpixel_pixel_map.size();
        int we = result->superpixelFeats_1.rows;
        int wf = result->image_idx_superpixel.size();
        int wg = result->superpixel_idx.size();
        int wh = result->image_rows.size();
        int wi = result->image_cols.size();
        */


        //Push back segments



        int a=10;



        std::cout<<"Feature/Patch Loaded : "<<count_data++<<std::endl;

        if(count_data>100)
         break;






      }


    }

    result->uniqueClasses_ = std::set<int> (result->labels_.begin(), result->labels_.end());
    result->dataPatches = true;


    return result;


  }

  /// <summary>
  /// NOTWORKING : Generate a 2D dataset with data points distributed in a grid pattern.
  /// Intended for generating visualization images.
  /// </summary>
  /// <param name="rangeX">x-axis range</param>
  /// <param name="nStepsX">Number of grid points in x direction</param>
  /// <param name="rangeY">y-axis range</param>
  /// <param name="nStepsY">Number of grid points in y direction</param>
  /// <returns>A new DataPointCollection</returns>

  std::auto_ptr<DataPointCollection> DataPointCollection::Generate2dGrid(
    std::pair<float, float> rangeX, int nStepsX,
    std::pair<float, float> rangeY, int nStepsY)
  {

    if (rangeX.first >= rangeX.second)
      throw std::runtime_error("Invalid x-axis range.");
    if (rangeY.first >= rangeY.second)
      throw std::runtime_error("Invalid y-axis range.");

    std::auto_ptr<DataPointCollection> result =  std::auto_ptr<DataPointCollection>(new DataPointCollection());

    result->dimension_ = 2;

    float stepX = (rangeX.second - rangeX.first) / nStepsX;
    float stepY = (rangeY.second - rangeY.first) / nStepsY;

    for (int j = 0; j < nStepsY; j++)
      for (int i = 0; i < nStepsX; i++)
      {
        //RPG-Removed
        //result->data_.push_back(rangeX.first + i * stepX);
        //result->data_.push_back(rangeY.first + j * stepY);
      }

      return result;
  }

  /// <summary>
  /// NOTWORKING - Generate a 1D dataset containing a given number of data points
  /// distributed at regular intervals within a given range. Intended for
  /// generating visualization images.
  /// </summary>
  /// <param name="range">Range</param>
  /// <param name="nStepsX">Number of grid points</param>
  /// <returns>A new DataPointCollection</returns>
  std::auto_ptr<DataPointCollection> DataPointCollection::Generate1dGrid(std::pair<float, float> range, int nSteps)
  {
    if (range.first >= range.second)
      throw std::runtime_error("Invalid range.");

    std::auto_ptr<DataPointCollection> result =  std::auto_ptr<DataPointCollection>(new DataPointCollection());

    result->dimension_ = 1;

    float step = (range.second - range.first) / nSteps;

    for (int i = 0; i < nSteps; i++)
    {
      //REMOVED - RPG
      //result->data_.push_back(range.first + i * step);
    }
    return result;
  }


  /// <summary>
  /// Get the data range in the specified data dimension.
  /// </summary>
  /// <param name="dimension">The dimension over which to compute min and max</param>
  /// <returns>A tuple containing min and max over the specified dimension of the data</returns>
  std::pair<float, float> DataPointCollection::GetRange(int dimension) const
  {
    if (Count() < 0)
      throw std::runtime_error("Insufficient data to compute range.");

    if (dimension < 0 || dimension>dimension_)
      throw std::runtime_error("Insufficient data to compute range.");

/*    float min = data_[0 + dimension], max = data_[0 + dimension];

    for (int i = 0; i < (int)(data_.size())/dimension_; i++)
    {
      if (data_[i*dimension_ +  dimension] < min)
        min = data_[i*dimension_ +  dimension];
      else if (data_[i*dimension_ +  dimension] > max)
        max = data_[i*dimension_ +  dimension];
    }
*/


    cv::Mat viewMat = dataMat.colRange(0, dimension);
    double min, max;
    cv::minMaxLoc(viewMat, &min, &max);
    return std::pair<float, float>((float)min, (float)max);
  }

  /// <summary>
  /// Get the range of target values (or raise an exception if these data
  /// do not have associated target values).
  /// </summary>
  /// <returns>A tuple containing the min and max target value for the data</returns>
  std::pair<float, float> DataPointCollection::GetTargetRange() const
  {
    if (!HasTargetValues())
      throw std::runtime_error("Data points do not have target values.");
    if (Count() < 0)
      throw std::runtime_error("Insufficient data to compute range.");

    float min = targets_[0], max = targets_[0];

    for (int i = 0; i < dataMat.cols; i++)
    {
      if (targets_[i] < min)
        min = targets_[i];
      else if (targets_[i] > max)
        max = targets_[i];
    }

    return std::pair<float, float>(min, max);
  }

  void tokenize(
    const std::string& str,
    std::vector<std::string>& tokens,
    const std::string& delimiters)
  {
    tokens.clear();

    std::string::size_type lastPos = str.find_first_not_of(delimiters, 0); // skip initial delimeters

    for(std::string::size_type i=0; i<(lastPos==std::string::npos?str.length()+1:lastPos); i++)
      tokens.push_back("");

    std::string::size_type pos = str.find_first_of(delimiters, lastPos); // first "non-delimiter"   

    while (std::string::npos != pos || std::string::npos != lastPos)
    {
      tokens.push_back(str.substr(lastPos, pos - lastPos)); // found token, add to vector 
      lastPos = str.find_first_not_of(delimiters, pos);     // skip delimiters

      if(pos!=std::string::npos)
        for(std::string::size_type i=pos+1; i<(lastPos==std::string::npos?str.length()+1:lastPos); i++)
          tokens.push_back("");

      pos = str.find_first_of(delimiters, lastPos);         // find next "non-delimiter"
    }
  }

  float to_float(const std::string& s)
  {
    std::istringstream ss(s);  
    float x;
    ss >> x;
    if(ss.fail())
      throw std::runtime_error("Failed to interpret number as floating point.");
    return x;
  }


  cv::Mat to_index_kitti(const cv::Mat &img)
  {

    for(int y=0;y<img.rows;y++)
      for(int x=0;x<img.cols;x++)
      {
        //std::cout<<img.at<cv::Vec3b>(y,x)<<std::endl;
        cv::Vec3b color =img.at<cv::Vec3b>(y,x);
        if(color == cv::Vec3b(0,0,255))
        {
          std::cout<<"No Road"<<std::endl;
        }
      }

    return cv::Mat::zeros(img.rows, img.cols, img.type());
  }








  //CAFFE stuff
  Classifier::Classifier(const string& model_file,
                         const string& trained_file,
                         const string& mean_file,
                         const string& label_file) {
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif

    std::cout<<"Load the network model file."<<std::endl;
    net_.reset(new Net<float>(model_file, TEST));
    std::cout<<"Load the network trained file."<<std::endl;
    net_->CopyTrainedLayersFrom(trained_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
    //input_geometry_ = cv::Size(512,512);

    /* Load the binaryproto mean file. */
    //std::cout<<"Not loading mean..."<<std::endl;

    std::cout<<"Load the Labels."<<std::endl;
    std::ifstream labels(label_file.c_str());
    CHECK(labels) << "Unable to open labels file " << label_file;
    string line;
    while (std::getline(labels, line))
      labels_.push_back(string(line));

    Blob<float>* output_layer = net_->output_blobs()[0];
    //CHECK_EQ(labels_.size(), output_layer->channels())
    //<< "Number of labels is different from the output layer dimension.";
  }






  cv::Mat Classifier::forwardPass(const cv::Mat &img, int layer_nr)
  {
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_geometry_ = img.size();
    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    //std::cout<<"Wrap input layer"<<std::endl;
    WrapInputLayer(&input_channels);
    //std::cout<<"Preprocess"<<std::endl;
    Preprocess(img, &input_channels);





    //std::cout<<"NetForward"<<std::endl;
    //net_->ForwardPrefilled();
    net_->ForwardTo(layer_nr);


    std::vector<std::string> layer_names = net_->blob_names();
    std::string blob_name = layer_names[layer_nr];

    boost::shared_ptr<Blob<float>> sample_blob;
    sample_blob = net_->blob_by_name(blob_name);
    //std::cout<<"Channels: "<<sample_blob->channels()<<std::endl;

    float* data = sample_blob->mutable_cpu_data();

    cv::Mat hypercolumns;
    std::vector<cv::Mat> channels;

    for (int i = 0; i < sample_blob->channels(); ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(sample_blob-> height(), sample_blob->width(), CV_32FC1, data);
      //channels.push_back(channel);
      data += sample_blob->height() * sample_blob->width();
      cv::Mat out;
      cv::resize(channel, out, img.size());

      channels.push_back(out);


      /*//Only to visualize
      cv::normalize(out, channel,255,0);
      cv::imshow("Channel", channel);
      cv::waitKey();*/

    }

    cv::merge(channels, hypercolumns);

    //std::cout<<hypercolumns.channels();

    return hypercolumns.clone();

  }


  std::vector<cv::Mat> Classifier::forwardPassVector(const cv::Mat &img, int layer_nr)
  {
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_geometry_ = img.size();
    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    //std::cout<<"Wrap input layer"<<std::endl;
    WrapInputLayer(&input_channels);
    //std::cout<<"Preprocess"<<std::endl;
    Preprocess(img, &input_channels);





    //std::cout<<"NetForward"<<std::endl;
    //net_->ForwardPrefilled();
    net_->ForwardTo(layer_nr);


    std::vector<std::string> layer_names = net_->blob_names();
    std::string blob_name = layer_names[layer_nr];

    boost::shared_ptr<Blob<float>> sample_blob;
    sample_blob = net_->blob_by_name(blob_name);
    //std::cout<<"Channels: "<<sample_blob->channels()<<std::endl;

    float* data = sample_blob->mutable_cpu_data();

    cv::Mat hypercolumns;
    std::vector<cv::Mat> channels;

    for (int i = 0; i < sample_blob->channels(); ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(sample_blob-> height(), sample_blob->width(), CV_32FC1, data);
      //channels.push_back(channel);
      data += sample_blob->height() * sample_blob->width();
      //cv::Mat out;
      //cv::resize(channel, out, img.size());

      channels.push_back(channel);


      /*//Only to visualize
      cv::normalize(out, channel,255,0);
      cv::imshow("Channel", channel);
      cv::waitKey();*/

    }

    std::cout<<channels[0].size();

    //cv::merge(channels, hypercolumns);

    //std::cout<<hypercolumns.channels();

    return channels;

  }









/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
  void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
      cv::Mat channel(height, width, CV_32FC1, input_data);
      input_channels->push_back(channel);
      input_data += width * height;
    }
  }

  void Classifier::Preprocess(const cv::Mat& img,
                              std::vector<cv::Mat>* input_channels) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
      cv::cvtColor(img, sample, CV_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
      cv::cvtColor(img, sample, CV_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
      cv::cvtColor(img, sample, CV_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
      cv::cvtColor(img, sample, CV_GRAY2BGR);
    else
      sample = img;

    cv::Mat sample_resized;
    //std::cout<<"....Resizing"<<std::endl;
    if (sample.size() != input_geometry_)
      cv::resize(sample, sample_resized, input_geometry_);
    else
      sample_resized = sample;

    cv::Mat sample_float;
    //std::cout<<"....Converting"<<std::endl;
    if (num_channels_ == 3)
      sample_resized.convertTo(sample_float, CV_32FC3);
    else
      sample_resized.convertTo(sample_float, CV_32FC1);


    cv::Mat sample_normalized;
    if(!mean_.empty())
      cv::subtract(sample_float, mean_, sample_normalized);
    else
      sample_normalized=sample_float.clone();

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
  }

        } } }
