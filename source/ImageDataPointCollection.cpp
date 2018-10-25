//
// Created by prassanna on 10/23/18.
//

#include "ImageDataPointCollection.h"
#include <fstream>
#include <sstream>
#include "nlohmann/json.hpp"


using namespace Slither;


std::vector<int> ImageDataPointCollection::tokenize_str_toint(std::string str)
{
    std::vector<int> vect;

    std::stringstream ss(str);

    int i;

    while (ss >> i)
    {
        vect.push_back(i);

        if (ss.peek() == ',')
            ss.ignore();
    }

    return vect;
}

std::unique_ptr<ImageDataPointCollection> ImageDataPointCollection::Load(const std::string &filename)
{

    std::unique_ptr<ImageDataPointCollection> result = std::unique_ptr<ImageDataPointCollection>(new ImageDataPointCollection());

    // Expecting a json file
    std::ifstream i(filename);
    nlohmann::json contents;
    i>>contents;

    if (contents.count("annotation_folder"))
        result->annotation_folder = contents["annotation_folder"];
    result->image_folder = contents["image_folder"];
    result->filenames  = std::vector<std::string> (contents["filenames"]);
    //std::string unique_labels = contents["labels"];
    std::vector<int> labels_int = result->tokenize_str_toint(contents["labels"]);
    result->uniqueClasses_ = std::set<int>(labels_int.begin(), labels_int.end());

    return result;
}