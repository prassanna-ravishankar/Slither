#include "DataPatchCollection.h"

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <opencv2/ximgproc.hpp>


namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{


  std::auto_ptr<DataPatchCollection> DataPatchCollection::Create(std::string folder_img, std::string folder_ann, cv::Size size)
  {
    std::auto_ptr<DataPatchCollection> result = std::auto_ptr<DataPatchCollection>(new DataPatchCollection());
    result->img_folder = folder_img;
    result->ann_folder = folder_ann;
    result->size_patches = size;

    return result;


  }


  void  DataPatchCollection::Load(const std::string &filename, std::auto_ptr<DataPatchCollection> &patches)
  {
    //cv::Size size(1242,375); //Todo :: Get as a parameter
    std::ifstream fin (filename.c_str());
    std::string delimiter = ":";
    std::string folder_name_images = filename.substr(0,filename.rfind("/")+1)+"Images/";

    std::cout<<"FOLDER is : "<<folder_name_images<<std::endl;


    int count_data=0;


    while(!fin.eof())
    {

      std::string labels;
      std::string line;
      std::getline(fin, line);
      int pos = 0;
      int lbl = 0;

      std::string image_name = patches->img_folder + line;
      std::string ann_name = patches->ann_folder + line;

      std::cout<<" On filename : "<<count_data++<<"\t-->"<<line;

      patches-> filenames.push_back(line);


      if(line.empty())
        continue;

      cv::Mat img = cv::imread(image_name);
      cv::resize(img, img, patches->size_patches);
      cv::Mat ann_gt = cv::imread(ann_name,CV_LOAD_IMAGE_GRAYSCALE);
      cv::resize(ann_gt, ann_gt, patches->size_patches);

      if(img.empty() || ann_gt.empty())
        continue;

      //For Kitti only
      cv::Mat ann  = cv::Mat(ann_gt.rows, ann_gt.cols, CV_8U);
      ann = ( ann_gt>=1) - 254;


      std::cout<<"\t--> Nr of superpixels ";

      for(int n=400;n<=1200;n=n+400)
      {
        float region_size = (int)std::sqrt((img.size().height*img.size().width) / n);
        cv::Mat labels;
        cv::Ptr<cv::ximgproc::SuperpixelSLIC>slic = cv::ximgproc::createSuperpixelSLIC(img, cv::ximgproc::SLIC, region_size,10.0f);
        slic->iterate();
        slic->enforceLabelConnectivity();
        slic->getLabels(labels);
        double minVal, maxVal;
        cv::minMaxIdx(labels,&minVal, &maxVal);

        if(n==400)
          patches->superpixel_scale_1.push_back(labels);
        if(n==800)
          patches->superpixel_scale_2.push_back(labels);
        if(n==1200)
          patches->superpixel_scale_3.push_back(labels);
        std::cout<<maxVal<<", ";
      }



      std::cout<<std::endl;
      //All the pushes.
      patches->images_.push_back(img);
      patches->annotations_.push_back(ann);
      patches->img_rows_.push_back(img.rows);
      patches->img_cols_.push_back(img.cols);

      uniqueElements(ann, patches->uniqueClasses_);
      //std::copy( unique_classes.begin(), unique_classes.end(), std::inserter( patches->uniqueClasses_, patches->uniqueClasses_.end() ) );

      if(patches->net_img_size_.empty())
        patches->net_img_size_.push_back(img.rows*img.cols);
      else
        patches->net_img_size_.push_back(patches->net_img_size_.back()+img.rows*img.cols);

      if(count_data>5)
        break;
      }


    patches->dimension_ = 64; //TODO :: For Now..... Change this into something dynamic
    patches->dataCount_ = patches->net_img_size_.back();


    //Initializing the network --> TODO :: model_files and train_files as parameters
    std::string model_file="/home/prassanna/Libraries/caffe-master/models/vggnet/VGG_CNN_F_copy.prototxt";
    std::string trained_file="/home/prassanna/Libraries/caffe-master/models/vggnet/VGG_CNN_F.caffemodel";

    std::auto_ptr<Classifier> bla(new Classifier(model_file,trained_file));
    patches->net_ =bla;



      //Trying out hypercolumns --- Testing....
      std::vector<int> layers_nr{1,2,35,40,16,13};
      cv::Point pt(10,10);
      cv::Mat temp_hypercolumn;
      patches->net_->forwardPassCollection(patches->images_, 3,layers_nr, patches->hypercolumns_pixel);
      patches->net_->forwardPassSingle(patches->images_[1], 3 , layers_nr, temp_hypercolumn);
      std::cout<<std::endl;

      std::vector<float> hypercolumn_calculated;
      //hypercolumn_calculated = patches->net_->getHypercolumnPoint(patches->images_[1], 3, layers_nr, pt);
      //hypercolumn_calculated = patches->net_->getHypercolumnPoint(patches->images_[1], 3, layers_nr, pt); //Error

      cv::Mat desc = patches->net_->getHypercolumnPoint(patches->images_[1], 3, layers_nr, pt);
      std::cout<<std::endl;

      //std::cout<<desc<<std::endl;
      desc = patches->net_->getHypercolumnPoint(patches->images_[1], 3, layers_nr, pt);
      std::cout<<std::endl;

      //std::cout<<desc<<std::endl;
      int sup_lbl = patches->net_->getSuperpixelLabel(patches->superpixel_scale_1[0],pt);
      std::cout<<std::endl;

      cv::Mat sup_desc = patches->net_->getSuperpixelDescriptor(patches->images_[1], patches->superpixel_scale_1[1], sup_lbl, 3, layers_nr);
      std::cout<<std::endl;

      //std::cout<<sup_desc<<std::endl;
      cv::Mat hcolumn = patches->GetDataPoint(0,3, layers_nr, true, true,false);
      std::cout<<hcolumn<<hcolumn.size()<<std::endl;
      hcolumn = patches->GetDataPoint(0,3, layers_nr, true, false,false);
      std::cout<<hcolumn<<hcolumn.size()<<std::endl;

      std::vector<unsigned int> bla2{0,1,2,3};

      patches->getTrainDataWithMaskHypercolumn(layers_nr, 3, 0,100,&bla2[0] );
      int a=5;


    }


    ///CAFFE STUFFF
    Classifier::Classifier()
    {

    }

    void Classifier::setNetworkSources(const std::string &model_file, const std::string &trained_file,
                                       const std::string &mean_file, const std::string &label_file)
    {
#ifdef CPU_ONLY
      caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
      Caffe::set_mode(Caffe::GPU);
#endif

      std::cout<<"Load the network model file."<<std::endl;
      net_.reset(new caffe::Net<float>(model_file, caffe::TEST));
      std::cout<<"Load the network trained file."<<std::endl;
      net_->CopyTrainedLayersFrom(trained_file);

      CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
      CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

      caffe::Blob<float>* input_layer = net_->input_blobs()[0];
      num_channels_ = input_layer->channels();
      CHECK(num_channels_ == 3 || num_channels_ == 1)
      << "Input layer should have 1 or 3 channels.";
      input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
      //input_geometry_ = cv::Size(512,512);

      /* Load the binaryproto mean file. */
      //std::cout<<"Not loading mean..."<<std::endl;

      std::cout<<"Load the Labels."<<std::endl;
      //std::ifstream labels(label_file.c_str());
      //CHECK(labels) << "Unable to open labels file " << label_file;
      //std::string line;
      //while (std::getline(labels, line))
     //   labels_.push_back(std::string(line));

      caffe::Blob<float>* output_layer = net_->output_blobs()[0];
      //CHECK_EQ(labels_.size(), output_layer->channels())
      //<< "Number of labels is different from the output layer dimension.";
    }

    Classifier::Classifier(const std::string& model_file,
                           const std::string& trained_file) {
#ifdef CPU_ONLY
      caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
      Caffe::set_mode(Caffe::GPU);
#endif

      std::cout<<"Load the network model file."<<std::endl;
      net_.reset(new caffe::Net<float>(model_file, caffe::TEST));
      std::cout<<"Load the network trained file."<<std::endl;
      net_->CopyTrainedLayersFrom(trained_file);

      CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
      CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

      caffe::Blob<float>* input_layer = net_->input_blobs()[0];
      num_channels_ = input_layer->channels();
      CHECK(num_channels_ == 3 || num_channels_ == 1)
      << "Input layer should have 1 or 3 channels.";
      input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
      //input_geometry_ = cv::Size(512,512);

      /* Load the binaryproto mean file. */
      //std::cout<<"Not loading mean..."<<std::endl;

      //std::cout<<"Load the Labels."<<std::endl;
        //std::ifstream labels(label_file.c_str());
      //CHECK(labels) << "Unable to open labels file " << label_file;
      //std::string line;
      //while (std::getline(labels, line))
        //labels_.push_back(std::string(line));

      caffe::Blob<float>* output_layer = net_->output_blobs()[0];
      //CHECK_EQ(labels_.size(), output_layer->channels())
      //<< "Number of labels is different from the output layer dimension.";
    }




    cv::Mat Classifier::forwardPass(const cv::Mat &img, int layer_nr)
    {
      caffe::Blob<float>* input_layer = net_->input_blobs()[0];
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

      boost::shared_ptr<caffe::Blob<float>> sample_blob;
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
      caffe::Blob<float>* input_layer = net_->input_blobs()[0];
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

      boost::shared_ptr<caffe::Blob<float>> sample_blob;
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

      //std::cout<<channels[0].size();

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
      caffe::Blob<float>* input_layer = net_->input_blobs()[0];

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


      cv::Mat sample_normalized=sample_float.clone();

      /* This operation will write the separate BGR planes directly to the
       * input layer of the network because it is wrapped by the cv::Mat
       * objects in input_channels. */
      cv::split(sample_normalized, *input_channels);

      CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
            == net_->input_blobs()[0]->cpu_data())
      << "Input channels are not wrapping the input layer of the network.";
    }


    void Classifier::forwardPassCollection(const std::vector<cv::Mat> &images, int layer_nr, std::vector<int>&channels_required, std::vector<cv::Mat> &result_vector)
    {
        for(int i=0;i<images.size();i++)
        {
            cv::Mat img = images[i];
            std::vector<cv::Mat> hypercolumn_channels = forwardPassVector(img, layer_nr);
            std::vector<cv::Mat> selected_channels;

            for(int c=0;c<channels_required.size();c++)
            {
                cv::Mat resized;
                cv::resize(hypercolumn_channels[channels_required[c]], resized, img.size());
                selected_channels.push_back(resized);
            }

            cv::Mat merged_channels;
            cv::merge(selected_channels,merged_channels);
            result_vector.push_back(merged_channels);
        }
    }


    void Classifier::forwardPassSingle(const cv::Mat &img, int layer_nr, std::vector<int>&channels_required, cv::Mat &results)
    {

        std::vector<cv::Mat> hypercolumn_channels = forwardPassVector(img, layer_nr);
        std::vector<cv::Mat> selected_channels;

        for(int c=0;c<channels_required.size();c++)
        {
            cv::Mat resized;
            cv::resize(hypercolumn_channels[channels_required[c]], resized, img.size());
            selected_channels.push_back(resized);
        }


        cv::merge(selected_channels,results);


    }


    std::vector<float> Classifier::calcHypercolumnPoint(const cv::Mat &img, int layer_nr, std::vector<int>&channels_required, cv::Point &location)
    {



        //When they are not equal, calculate the hypercolumn manually and cache it
        cached_image = img.clone();
        forwardPassSingle(img, layer_nr, channels_required, cached_hypercolumn);
        //std::cout<<"Manual operation"<<std::endl;

        std::vector<float> hypercolumn;

        if(cached_hypercolumn.channels()>1)
        {
            std::vector<cv::Mat> channels;
            cv::split(cached_hypercolumn, channels);
            for(int i = 0;i<channels.size();i++)
            {
                hypercolumn.push_back(channels[i].at<float>(location));

            }

        }
        else
            hypercolumn.push_back(cached_hypercolumn.at<float>(location));


        return hypercolumn;

    }


    std::vector<float> Classifier::cachedHypercolumnPoint(int layer_nr, std::vector<int>&channels_required, cv::Point &location)
    {

        std::vector<float> hypercolumn;
        if(cached_hypercolumn.channels()>1)
        {
            std::vector<cv::Mat> channels;
            cv::split(cached_hypercolumn, channels);
            for(int i = 0;i<channels.size();i++)
            {
                hypercolumn.push_back(channels[i].at<float>(location));

            }

        }
        else
            hypercolumn.push_back(cached_hypercolumn.at<float>(location));


        return hypercolumn;

    }

    cv::Mat Classifier::getHypercolumnPoint(const cv::Mat &img, int layer_nr, std::vector<int>&channels_required, cv::Point &location)
    {
        std::vector<float> hypercolumns;
        if(img.empty())
            hypercolumns =  calcHypercolumnPoint(img, layer_nr, channels_required, location);
        else
        {
            if(matIsEqual(cached_image, img))
                hypercolumns = cachedHypercolumnPoint(layer_nr, channels_required, location);
            else
                hypercolumns = calcHypercolumnPoint(img, layer_nr, channels_required, location);
        }

        cv::Mat hcolumn = cv::Mat::zeros(1, hypercolumns.size(), CV_32FC1);

        for(int i=0;i<hypercolumns.size();i++)
        {
            hcolumn.at<float>(0,i) = hypercolumns[i];
        }

        return hcolumn.clone();

    }

    int Classifier::getSuperpixelLabel(const cv::Mat &img_labels, cv::Point &location)
    {

        return img_labels.at<int>(location);
    }

    std::vector<cv::Point> Classifier::getSuperpixelPoints(const cv::Mat &img_labels, int lbl)
    {

        std::vector<cv::Point> p;
        int temp_index = 0;
        for(int y=0;y<img_labels.rows;y++)
            for(int x=0;x<img_labels.cols;x++)
            {
                temp_index = img_labels.at<int>(y,x);
                if(temp_index==lbl)
                {
                    cv::Point pt(x,y);
                    p.push_back(pt);
                }
            }
        return p;

    }

    cv::Mat Classifier::getSuperpixelDescriptor(const cv::Mat &img, const cv::Mat &img_labels, int lbl, int layer_nr, std::vector<int> &channels_required)
    {
        std::vector<cv::Point> pts = getSuperpixelPoints(img_labels, lbl);
        std::vector<float> hypercolumn_temp = getHypercolumnPoint(img,layer_nr, channels_required, pts[0]);
        cv::Mat descriptor= cv::Mat::zeros(1,hypercolumn_temp.size(), CV_32FC1);
        for(int i = 0;i<pts.size();i++)
        {
            std::vector<float> hypercolumn = getHypercolumnPoint(img,layer_nr, channels_required, pts[0]);
            for(int c = 0;c<hypercolumn.size();c++)
            {
                descriptor.at<float>(0,c) += hypercolumn[c];
            }
        }

        descriptor = descriptor / pts.size();

        return descriptor.clone();
    }








        } } }
