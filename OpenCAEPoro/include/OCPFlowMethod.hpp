/*! \file    OCPFlowMethod.hpp
 *  \brief   OCPFlowMethod class declaration
 *  \author  Shizhe Li
 *  \date    Oct/04/2023
 *
 *-----------------------------------------------------------------------------------
 *  Copyright (C) 2021--present by the OpenCAEPoroX team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *-----------------------------------------------------------------------------------
 */

#ifndef __OCPFLOWMETHOD_HEADER__
#define __OCPFLOWMETHOD_HEADER__

#include "OCPFlowVarSet.hpp"
#include "OCPFuncSAT.hpp"
#include <vector>

using namespace std;


class OCPFlowMethod
{
public:
    OCPFlowMethod() = default;
    /// calculate relative permeability and capillary pressure
    virtual void CalKrPc(OCPFlowVarSet& vs) = 0;
    /// calculate relative permeability and capillary pressure and derivatives
    virtual void CalKrPcDer(OCPFlowVarSet& vs) = 0;
    /// get saturation of connate water
    virtual OCP_DBL GetSwco() const = 0;
    /// get maximum capillary pressure between water and oil (Po - Pw)
    virtual OCP_DBL GetMaxPcow() const = 0;
    /// get minimum capillary pressure between water and oil (Po - Pw)
    virtual OCP_DBL GetMinPcow() const = 0;
    /// calculate Pcow by Sw
    virtual OCP_DBL CalPcowBySw(const OCP_DBL& Sw) const = 0;
    /// calculate Sw by Pcow 
    virtual OCP_DBL CalSwByPcow(const OCP_DBL& Pcow) const = 0;
    /// calculate Pcgo by Sg
    virtual OCP_DBL CalPcgoBySg(const OCP_DBL& Sg) const = 0;
    /// calculate Sg by Pcgo
    virtual OCP_DBL CalSgByPcgo(const OCP_DBL& Pcgo) const = 0;
    /// calculate Sw by Pcgw
    virtual OCP_DBL CalSwByPcgw(const OCP_DBL& Pcgw) const = 0;
    /// calculate Krg and ders by Sg
    virtual OCP_DBL CalKrg(const OCP_DBL& Sg, OCP_DBL& dKrgdSg) const = 0;
};


/////////////////////////////////////////////////////
// OCPFlowMethod_OGW01
/////////////////////////////////////////////////////


/// Use SGOF, SWOF
class OCPFlowMethod_OGW01 : public OCPFlowMethod
{
public:
    OCPFlowMethod_OGW01(const vector<vector<OCP_DBL>>& SGOFin,
        const vector<vector<OCP_DBL>>& SWOFin,
        const USI& i, OCPFlowVarSet& vs);
    void CalKrPc(OCPFlowVarSet& vs) override;
    void CalKrPcDer(OCPFlowVarSet& vs) override;
    OCP_DBL GetSwco() const override { return SWOF.GetSwco(); }
    OCP_DBL GetMaxPcow() const override { return SWOF.GetMaxPc(); }
    OCP_DBL GetMinPcow() const override { return SWOF.GetMinPc(); }
    OCP_DBL CalPcowBySw(const OCP_DBL& Sw) const override { return SWOF.CalPcow(Sw); }
    OCP_DBL CalSwByPcow(const OCP_DBL& Pcow) const override { return SWOF.CalSw(Pcow); }
    OCP_DBL CalPcgoBySg(const OCP_DBL& Sg) const override { return SGOF.CalPcgo(Sg); }
    OCP_DBL CalSgByPcgo(const OCP_DBL& Pcgo) const override { return SGOF.CalSg(Pcgo); }
    OCP_DBL CalSwByPcgw(const OCP_DBL& Pcgw) const override { return SWPCGW.Eval_Inv(1, Pcgw, 0); }
    OCP_DBL CalKrg(const OCP_DBL& Sg, OCP_DBL& dKrgdSg) const override { return SGOF.CalKrg(Sg, dKrgdSg); }

protected:
    void Generate_SWPCWG();

protected:
    OCP_SGOF                SGOF;
    OCP_SWOF                SWOF;
    /// auxiliary table: saturation of water vs. Pcgw
    OCPTable                SWPCGW;

protected:
    OCP3POilPerCalculation  opC;
};


/////////////////////////////////////////////////////
// OCPFlowMethod_OGW02
/////////////////////////////////////////////////////


/// Use SOF3, SGFN, SWFN
class OCPFlowMethod_OGW02 : public OCPFlowMethod
{
public:
    OCPFlowMethod_OGW02(const vector<vector<OCP_DBL>>& SOF3in,
        const vector<vector<OCP_DBL>>& SGFNin,
        const vector<vector<OCP_DBL>>& SWFNin,
        const USI& i, OCPFlowVarSet& vs);
    void CalKrPc(OCPFlowVarSet& vs) override;
    void CalKrPcDer(OCPFlowVarSet& vs) override;

    OCP_DBL GetSwco() const override { return SWFN.GetSwco(); }
    OCP_DBL GetMaxPcow() const override { return SWFN.GetMaxPc(); }
    OCP_DBL GetMinPcow() const override { return SWFN.GetMinPc(); }

    OCP_DBL CalPcowBySw(const OCP_DBL& Sw) const override { return SWFN.CalPcow(Sw); }
    OCP_DBL CalSwByPcow(const OCP_DBL& Pcow) const override { return SWFN.CalSw(Pcow); }
    OCP_DBL CalPcgoBySg(const OCP_DBL& Sg) const override { return SGFN.CalPcgo(Sg); }
    OCP_DBL CalSgByPcgo(const OCP_DBL& Pcgo) const override { return SGFN.CalSg(Pcgo); }
    OCP_DBL CalSwByPcgw(const OCP_DBL& Pcgw) const override { return SWPCGW.Eval_Inv(1, Pcgw, 0); }
    OCP_DBL CalKrg(const OCP_DBL& Sg, OCP_DBL& dKrgdSg) const override { return SGFN.CalKrg(Sg, dKrgdSg); }

protected:
    void Generate_SWPCWG();

protected:
    OCP_SOF3            SOF3;
    OCP_SGFN            SGFN;
    OCP_SWFN            SWFN;
    /// auxiliary table: saturation of water vs. Pcgw
    OCPTable            SWPCGW; 


protected:
    OCP3POilPerCalculation  opC;
};


/////////////////////////////////////////////////////
// OCPFlowMethod_OW01
/////////////////////////////////////////////////////


/// Use SWOF
class OCPFlowMethod_OW01 : public OCPFlowMethod
{
public:
    OCPFlowMethod_OW01(const vector<vector<OCP_DBL>>& SWOFin, OCPFlowVarSet& vs);
    void CalKrPc(OCPFlowVarSet& vs) override;
    void CalKrPcDer(OCPFlowVarSet& vs) override;
    OCP_DBL GetSwco() const override { return SWOF.GetSwco(); }
    OCP_DBL GetMaxPcow() const override { return SWOF.GetMaxPc(); }
    OCP_DBL GetMinPcow() const override { return SWOF.GetMinPc(); }
    OCP_DBL CalPcowBySw(const OCP_DBL& Sw) const override { return SWOF.CalPcow(Sw); }
    OCP_DBL CalSwByPcow(const OCP_DBL& Pcow) const override { return SWOF.CalSw(Pcow); }
    OCP_DBL CalPcgoBySg(const OCP_DBL& Sg) const  override { OCP_ABORT("Inavailable!"); }
    OCP_DBL CalSgByPcgo(const OCP_DBL& Pcgo) const override { OCP_ABORT("Inavailable!"); }
    OCP_DBL CalSwByPcgw(const OCP_DBL& Pcgw) const override { OCP_ABORT("Inavailable!"); }
    OCP_DBL CalKrg(const OCP_DBL& Sg, OCP_DBL& dKrgdSg) const override { OCP_ABORT("Inavailable!"); }

protected:
    OCP_SWOF            SWOF;
};


/////////////////////////////////////////////////////
// OCPFlowMethod_OG01
/////////////////////////////////////////////////////


/// Use SGOF
class OCPFlowMethod_OG01 : public OCPFlowMethod
{
public:
    OCPFlowMethod_OG01(const vector<vector<OCP_DBL>>& SGOFin, OCPFlowVarSet& vs);
    void CalKrPc(OCPFlowVarSet& vs) override;
    void CalKrPcDer(OCPFlowVarSet& vs) override;
    OCP_DBL GetSwco() const override { OCP_ABORT("Inavailable!"); }
    OCP_DBL GetMaxPcow() const override { OCP_ABORT("Inavailable!"); }
    OCP_DBL GetMinPcow() const override { OCP_ABORT("Inavailable!"); }
    OCP_DBL CalPcowBySw(const OCP_DBL& Sw) const override { OCP_ABORT("Inavailable!"); }
    OCP_DBL CalSwByPcow(const OCP_DBL& Pcow) const override { OCP_ABORT("Inavailable!"); }
    OCP_DBL CalPcgoBySg(const OCP_DBL& Sg) const override { return SGOF.CalPcgo(Sg); }
    OCP_DBL CalSgByPcgo(const OCP_DBL& Pcgo) const override { return SGOF.CalSg(Pcgo); }
    OCP_DBL CalSwByPcgw(const OCP_DBL& Pcgw) const override { OCP_ABORT("Inavailable!"); }
    OCP_DBL CalKrg(const OCP_DBL& Sg, OCP_DBL& dKrgdSg) const override { return SGOF.CalKrg(Sg, dKrgdSg); }

protected:
    OCP_SGOF            SGOF;
};


/////////////////////////////////////////////////////
// OCPFlowMethod_GW01
/////////////////////////////////////////////////////


/// Use Brooks-Corey type model
class OCPFlowMethod_GW01 : public OCPFlowMethod
{
public:
    OCPFlowMethod_GW01(const BrooksCoreyParam& bcp, OCPFlowVarSet& vs) {
        vs.Init(OCPFlowType::GW, 2, 2);
        bc.Setup(bcp);
    }
    void CalKrPc(OCPFlowVarSet& vs) override;
    void CalKrPcDer(OCPFlowVarSet& vs) override;
    OCP_DBL GetSwco() const override { OCP_ABORT("Inavailable!"); }
    OCP_DBL GetMaxPcow() const override { OCP_ABORT("Inavailable!"); }
    OCP_DBL GetMinPcow() const override { OCP_ABORT("Inavailable!"); }
    OCP_DBL CalPcowBySw(const OCP_DBL& Sw) const override { OCP_ABORT("Inavailable!"); }
    OCP_DBL CalSwByPcow(const OCP_DBL& Pcow) const override { OCP_ABORT("Inavailable!"); }
    OCP_DBL CalPcgoBySg(const OCP_DBL& Sg) const override { OCP_ABORT("Inavailable!"); }
    OCP_DBL CalSgByPcgo(const OCP_DBL& Pcgo) const override { OCP_ABORT("Inavailable!"); }
    OCP_DBL CalSwByPcgw(const OCP_DBL& Pcgw) const override { OCP_ABORT("Inavailable!"); }
    OCP_DBL CalKrg(const OCP_DBL& Sg, OCP_DBL& dKrgdSg) const override { OCP_ABORT("Inavailable!"); }

protected:
    BrooksCorey   bc;
};


#endif /* end if __OCPFLOWMETHOD_HEADER__ */

/*----------------------------------------------------------------------------*/
/*  Brief Change History of This File                                         */
/*----------------------------------------------------------------------------*/
/*  Author              Date             Actions                              */
/*----------------------------------------------------------------------------*/
/*  Shizhe Li           Oct/04/2023      Create file                          */
/*----------------------------------------------------------------------------*/