#include <cassert>
#include <iostream>
#include "DataPointCollection.h"

int main() {
    auto data = Slither::DataPointCollection::Load("../data/sclf/sample_train.txt");
    if(!data) {
        std::cerr << "Failed to load dataset" << std::endl;
        return 1;
    }
    assert(data->Count() == 180);
    assert(data->Dimensions() == 64);
    auto row = data->GetDataPoint(0);
    assert(row.size() == 64);
    std::cout << "DataPointCollection tests passed" << std::endl;
    return 0;
}
