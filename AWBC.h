#ifndef ZHU_PPGENERALLOCAL_H
#define ZHU_PPGENERALLOCAL_H

#pragma once
#include "incrementalLearner.h"
#include "xxyDist.h"
#include <limits>
#include "crosstab.h"
#include "distributionTree.h"
#include "yDist.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include "instanceStream.h"
#include "utils.h"
#include "StoredInstanceStream.h"
#include <assert.h>
#include "xxxxyDist.h"

class zhu_ppgenerallocal:public IncrementalLearner
{
    public:
        zhu_ppgenerallocal();
        zhu_ppgenerallocal(char* const *& argv,char* const * end );
        virtual ~zhu_ppgenerallocal();

        void reset(InstanceStream &is);
        void initialisePass();
        void train(const instance &inst);
        void finalisePass();
        bool trainingIsFinished();
        void getCapabilities(capabilities &c);

        void printVector(std::vector<CategoricalAttribute>  &temp_vector);
        virtual void classify(const instance &inst, std::vector<double> &classDist);


    protected:

    private:
        unsigned int noCatAtts_;
        unsigned int noClasses_;

        InstanceStream* instanceStream_;
        std::vector<CategoricalAttribute> parents_;

        xxyDist Dist;

        //StoredInstanceStream  StoredInstanceStream_;

        //std::vector<std::vector<InstanceCount> > counts_;
        bool trainingIsFinished_;
        const static CategoricalAttribute NOPARENT = 0xFFFFFFFFUL;
};

#endif



