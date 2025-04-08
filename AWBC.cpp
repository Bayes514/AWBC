#include "zhu_ppgenerallocal.h"


#include "utils.h"
#include "correlationMeasures.h"
#include <assert.h>
#include <math.h>
#include <set>
#include <stdlib.h>
#include <iostream>


#include "ALGLIB_specialfunctions.h"

using namespace std;


zhu_ppgenerallocal::zhu_ppgenerallocal() :
trainingIsFinished_(false)
{
}

zhu_ppgenerallocal::zhu_ppgenerallocal(char* const *&, char* const *) :
trainingIsFinished_(false)
{
    name_ = "zhu_ppgenerallocal";
}

zhu_ppgenerallocal::~zhu_ppgenerallocal()
{
    //析构函数
}
void zhu_ppgenerallocal::reset(InstanceStream &is)
{

    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();

    trainingIsFinished_ = false;

    parents_.resize(noCatAtts_);
   Dist.reset(is);

}

void zhu_ppgenerallocal::getCapabilities(capabilities &c)
{
    c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void zhu_ppgenerallocal::initialisePass()
{
    assert(trainingIsFinished_ == false);

}
void zhu_ppgenerallocal::train(const instance &inst)
{
    Dist.update(inst);


}

void zhu_ppgenerallocal::finalisePass()
{
    assert(trainingIsFinished_ == false);

    // printf("---------classify-------\n");

    //存放已经进入结构的节点的容器
    std::vector<CategoricalAttribute> order;

    //存放还未进入结构的节点的容器
    std::vector<CategoricalAttribute> notorder;
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++)
    {
        notorder.push_back(a);
       // parentsLocal[a] = NOPARENT;
    }

    //找第一个进入网络的节点
    double max_first =  -std::numeric_limits<double>::max();
    CategoricalAttribute first_a;
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++)
        {
             double pp_first = 0.0;
             for (CatValue v = 0; v < Dist.getNoValues(a); v++)
                {
                for (CatValue y = 0; y < noClasses_; y++)
                    {
                        pp_first += Dist.xyCounts.jointP(a,v,y) * Dist.xyCounts.jointP(a,v,y);
                    }
                }
            if (pp_first > max_first)
            {
                max_first = pp_first;
                first_a = a;
            }
        }
     //printf("找到的第一个节点： %d \n", first_a);

    //将第一个节点加入order容器中  //将节点从还未进入的容器中删除
      order.push_back(first_a);
      parents_[first_a] = NOPARENT;
      notorder.erase(std::remove(notorder.begin(), notorder.end(), first_a), notorder.end());

    //找第接下来进入网络的节点
      while(!notorder.empty())
    {
        double max_pp =  -std::numeric_limits<double>::max();
        CategoricalAttribute next_a;
        CategoricalAttribute next_a_pa =  NOPARENT;
        for (std::vector<CategoricalAttribute>::const_iterator it = notorder.begin() ; it != notorder.end(); it++)
      {
          CategoricalAttribute it_pa =  NOPARENT;
          //记录一个节点内的最大值
          double pp = -std::numeric_limits<double>::max();
          for (std::vector<CategoricalAttribute>::const_iterator it2 = order.begin() ; it2 != order.end(); it2++)
        {
            double ppde = 0.0;
            double ppud = 0.0;
            for (CatValue vit = 0; vit < Dist.getNoValues(*it); vit++)
            {
                for (CatValue vit2 = 0; vit2 < Dist.getNoValues(*it2); vit2++)
                {
                    for (CatValue y = 0; y < noClasses_; y++)
                    {

                        ppde += Dist.jointP(*it2,vit2,*it,vit,y) * Dist.jointP(*it2,vit2,*it,vit,y);
                        ppud += Dist.jointP(*it2,vit2,*it,vit,y) * Dist.xyCounts.p(*it2,vit2,y)* Dist.xyCounts.p(*it,vit,y) *Dist.xyCounts.p(y);
                    }
                }
            }

             if (ppud >= ppde)
            {
                printf("-------------Tp<0-----------------\n");
                pp = ppud;
                it_pa = NOPARENT;
            }
            else
            {
                pp = ppde;
                it_pa= *it2;
            }
        }

        if (pp > max_pp)
            {
                max_pp = pp;
                next_a = *it;
                next_a_pa = it_pa;
            }
      }
     //printf("加入网络中下一个X是：%d ,父节点是：%d , max_pp值是 ： %f \n",next_a,next_a_pa,max_pp);
      //将下一个节点加入order容器中  //将节点从还未进入的容器中删除
      order.push_back(next_a);
      parents_[next_a] = next_a_pa;
      notorder.erase(std::remove(notorder.begin(), notorder.end(), next_a), notorder.end());

    }


    trainingIsFinished_ = true;
}

bool zhu_ppgenerallocal::trainingIsFinished()
{
    return trainingIsFinished_;
}




void zhu_ppgenerallocal::classify(const instance &inst, std::vector<double> &classDist)
{
    //local zhu_pp
    //printf("---------classify-------\n");
    std::vector<CategoricalAttribute> parentsLocal;
    parentsLocal.resize(noCatAtts_);

    //存放已经进入结构的节点的容器
    std::vector<CategoricalAttribute> orderloc;

    //存放还未进入结构的节点的容器
    std::vector<CategoricalAttribute> notorderloc;
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++)
    {
        notorderloc.push_back(a);
       // parentsLocal[a] = NOPARENT;
    }

    //找第一个进入网络的节点
    double max_first =  -std::numeric_limits<double>::max();
    CategoricalAttribute first_a;
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++)
        {
             double pp_first = 0.0;
             for (CatValue y = 0; y < noClasses_; y++)
                {
                    pp_first += Dist.xyCounts.jointP(a,inst.getCatVal(a),y) * Dist.xyCounts.jointP(a,inst.getCatVal(a),y);
                }
            if (pp_first > max_first)
            {
                max_first = pp_first;
                first_a = a;
            }
        }
     //printf("找到的第一个节点： %d \n", first_a);

    //将第一个节点加入order容器中  //将节点从还未进入的容器中删除
      orderloc.push_back(first_a);
      parentsLocal[first_a] = NOPARENT;
      notorderloc.erase(std::remove(notorderloc.begin(), notorderloc.end(), first_a), notorderloc.end());

    //找第接下来进入网络的节点
      while(!notorderloc.empty())
    {
        double max_pp =  -std::numeric_limits<double>::max();
        CategoricalAttribute next_a;
        CategoricalAttribute next_a_pa =  NOPARENT;
        for (std::vector<CategoricalAttribute>::const_iterator it = notorderloc.begin() ; it != notorderloc.end(); it++)
      {
          CategoricalAttribute it_pa =  NOPARENT;
          //记录一个节点内的最大值
          double pp = -std::numeric_limits<double>::max();
          for (std::vector<CategoricalAttribute>::const_iterator it2 = orderloc.begin() ; it2 != orderloc.end(); it2++)
        {
            double ppde = 0.0;
            double ppud = 0.0;
            for (CatValue y = 0; y < noClasses_; y++)
                {

                    ppde += Dist.jointP(*it2,inst.getCatVal(*it2),*it,inst.getCatVal(*it),y) * Dist.jointP(*it2,inst.getCatVal(*it2),*it,inst.getCatVal(*it),y);
                    ppud += Dist.jointP(*it2,inst.getCatVal(*it2),*it,inst.getCatVal(*it),y) * Dist.xyCounts.p(*it2,inst.getCatVal(*it2),y)* Dist.xyCounts.p(*it,inst.getCatVal(*it),y) *Dist.xyCounts.p(y);
                }

             if (ppud >= ppde)
            {
                pp = ppud;
                it_pa = NOPARENT;
            }
            else
            {
                pp = ppde;
                it_pa= *it2;
            }

            /*
            if (ppud > pp)
            {
                pp = ppud;
                it_pa = NOPARENT;
            }
            if(ppde > pp)
            {
                pp = ppde;
                it_pa= *it2;
            }
            */
            //printf("对于未加入的属性 %d , 与 %d 独立时ppud= %f, 相关时ppde = %f \n",*it,*it2,ppud,ppde);
        }

        if (pp > max_pp)
            {
                max_pp = pp;
                next_a = *it;
                next_a_pa = it_pa;
            }
      }
      //printf("加入网络中下一个X是：%d ,父节点是：%d , max_pp值是 ： %f \n",next_a,next_a_pa,max_pp);
      //将下一个节点加入order容器中  //将节点从还未进入的容器中删除
      orderloc.push_back(next_a);
      parentsLocal[next_a] = next_a_pa;
      notorderloc.erase(std::remove(notorderloc.begin(), notorderloc.end(), next_a), notorderloc.end());

    }
    for (CatValue y = 0; y < noClasses_; y++)
    {
        classDist[y] = Dist.xyCounts.p(y);
    }

    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++)
    {
        const CategoricalAttribute parent = parentsLocal[x1];

        if (parent == NOPARENT)
        {
            for (CatValue y = 0; y < noClasses_; y++)
            {
                classDist[y] *= Dist.xyCounts.p(x1, inst.getCatVal(x1), y);
            }
        } else
        {
            for (CatValue y = 0; y < noClasses_; y++)
            {
                classDist[y] *= Dist.p(x1, inst.getCatVal(x1), parent,
                        inst.getCatVal(parent), y);
            }
        }
    }

    normalise(classDist);


    //general zhu_pp
    std::vector<double> classDistGeneral;
    classDistGeneral.assign(noClasses_, 0);

    for (CatValue y = 0; y < noClasses_; y++)
    {
        classDistGeneral[y] = Dist.xyCounts.p(y);
    }

    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++)
    {
        const CategoricalAttribute parent = parents_[x1];

        if (parent == NOPARENT)
        {
            for (CatValue y = 0; y < noClasses_; y++)
            {
                classDistGeneral[y] *= Dist.xyCounts.p(x1, inst.getCatVal(x1), y);
            }
        } else
        {
            for (CatValue y = 0; y < noClasses_; y++)
            {
                classDistGeneral[y] *= Dist.p(x1, inst.getCatVal(x1), parent,
                        inst.getCatVal(parent), y);
            }
        }
    }

    normalise(classDistGeneral);

    //联合概率结合
        for (int classno = 0; classno < noClasses_; classno++)
        {
            classDist[classno] += classDistGeneral[classno];
            classDist[classno] = classDist[classno] / 2;
        }

        //printf("\n the value of classDist[y] is: \n %f \n", classDist[0]);

}






