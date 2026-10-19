"""Fixed endocardial branch of the 43-state MATLAB model_SAC RHS.

``gating_intermediates`` exposes currents for MATLAB parity audits and the
assembled ``deriv`` tensor for integration. It currently supports the native
constant-parameter SAC law; the heterogeneity and alternate SAC forms in
human_bidirectional_rhs.m require separate handling.
"""
import torch


def gating_intermediates(x, t=0.0, lam=1.0, par=(0.2203, 10.3203, 12.8208, 1.4110, 0.8742),
                         isac_factor=(4.47, 0.001248), return_intermediates=True):
    x=torch.as_tensor(x)
    v,nai,nass,ki,kss,cai,cass,cansr,cajsr,m,hp,h,j,jp,mL,hL,hLp,a,iF,iS,ap,iFp,iSp = [x[...,i] for i in range(23)]
    jca=x[...,28]; nca,nca_i=x[...,29],x[...,30]
    CaMKt=x[...,36]
    R,T,F=8314.,310.,96485.; nao,cao,ko=140.,1.8,5.
    KmCaMK,aCaMK,bCaMK,CaMKo,KmCaM=.15,.05,.00068,.05,.0015
    CaMKb=CaMKo*(1-CaMKt)/(1+KmCaM/cass); CaMKa=CaMKb+CaMKt
    dCaMKt=aCaMK*CaMKb*(CaMKb+CaMKt)-bCaMK*CaMKt
    ENa=(R*T/F)*torch.log(nao/nai); EK=(R*T/F)*torch.log(ko/ki)
    PKNa=.01833; EKs=(R*T/F)*torch.log((ko+PKNa*nao)/(ki+PKNa*nai))
    vffrt=v*F*F/(R*T); vfrt=v*F/(R*T)
    fINap=1/(1+KmCaMK/CaMKa); fINaLp=fINap

    stretch_d=torch.as_tensor(lam,dtype=x.dtype,device=x.device)-1
    dgate=x[...,23]
    sig=stretch_d**isac_factor[0]/(stretch_d**isac_factor[0]+isac_factor[1])
    BetaNs=par[0]*10.; BetaKo=par[4]; Gns=par[1]*1e-6*sig; GKo=par[2]*1e-6; r=par[3]
    Cm=((stretch_d**4)/(.035**4+stretch_d**4)+1)*(1.534e-4)
    GammaSLNs=BetaNs*((2*stretch_d**3)/(.15**3+stretch_d**3))*.25
    INsNa=r*Gns*GammaSLNs*(v-ENa)/Cm; INsK=Gns*GammaSLNs*(v-EK)/Cm; INs=INsNa+INsK
    GammaSLKo=BetaKo*stretch_d+.7
    IKo=(GKo*(GammaSLKo/(1+torch.exp(-(10+v)/45)))*(v-EK)/Cm)/35.

    mss=1/((1+torch.exp(-(56.86+v)/9.03))**2)
    taum=.1292*torch.exp(-((v+45.79)/15.54)**2)+.06487*torch.exp(-((v-4.823)/51.12)**2)
    dm=(mss-m)/taum
    ah=torch.where(v>=-40,torch.zeros_like(v),.057*torch.exp(-(v+80)/6.8))
    bh=torch.where(v>=-40,.77/(.13*(1+torch.exp(-(v+10.66)/11.1))),2.7*torch.exp(.079*v)+3.1e5*torch.exp(.3485*v))
    tauh=1/(ah+bh); hss=1/((1+torch.exp((v+71.55)/7.43))**2); dh=(hss-h)/tauh
    aj=torch.where(v>=-40,torch.zeros_like(v),((-2.5428e4*torch.exp(.2444*v)-6.948e-6*torch.exp(-.04391*v))*(v+37.78))/(1+torch.exp(.311*(v+79.23))))
    bj=torch.where(v>=-40,.6*torch.exp(.057*v)/(1+torch.exp(-.1*(v+32))),.02424*torch.exp(-.01052*v)/(1+torch.exp(-.1378*(v+40.14))))
    tauj=1/(aj+bj); jss=1/((1+torch.exp((v+71.55)/7.43))**2); dj=(jss-j)/tauj
    hssp=1/((1+torch.exp((v+71.55+6)/7.43))**2); dhp=(hssp-hp)/tauh; djp=(jss-jp)/(1.46*tauj)
    INa=11.7802*(v-ENa)*m**3*((1-fINap)*h*j+fINap*hp*jp)

    mLss=1/(1+torch.exp(-(v+42.85)/5.264)); dmL=(mLss-mL)/taum
    hLss=1/(1+torch.exp((v+87.61)/7.488)); dhL=(hLss-hL)/200.
    hLssp=1/(1+torch.exp((v+93.81)/7.488)); dhLp=(hLssp-hLp)/600.
    INaL=.0279*(v-ENa)*mL*((1-fINaLp)*hL+fINaLp*hLp)

    # Ito activation/inactivation and phosphorylated branch.
    ass=1/(1+torch.exp(-(v-14.34)/14.82))
    ta=1.0515/(1/(1.2089*(1+torch.exp(-(v-18.4099)/29.3814)))+3.5/(1+torch.exp((v+100)/29.3814)))
    da=(ass-a)/ta
    iss=1/(1+torch.exp((v+43.94)/5.711))
    tiF=4.562+1/(.3933*torch.exp(-(v+100)/100)+.08004*torch.exp((v+50)/16.59))
    tiS=23.62+1/(.001416*torch.exp(-(v+96.52)/59.05)+1.780e-8*torch.exp((v+114.1)/8.079))
    diF=(iss-iF)/tiF; diS=(iss-iS)/tiS
    AiF=1/(1+torch.exp((v-213.6)/151.2)); AiS=1-AiF; i=AiF*iF+AiS*iS
    assp=1/(1+torch.exp(-(v-24.34)/14.82)); dap=(assp-ap)/ta
    dti_develop=1.354+0.0001/(torch.exp((v-167.4)/15.89)+torch.exp(-(v-12.23)/0.2154))
    dti_recover=1-.5/(1+torch.exp((v+70)/20))
    diFp=(iss-iFp)/(dti_develop*dti_recover*tiF)
    diSp=(iss-iSp)/(dti_develop*dti_recover*tiS)
    ip=AiF*iFp+AiS*iSp
    Ito=.16*(v-EK)*((1-fINaLp)*a*i+fINaLp*ap*ip)

    # ICaL gate kinetics (currents themselves are assembled in the next port).
    dss=1.0763*torch.exp(-1.0070*torch.exp(-.0829*v))
    dss=torch.where(v>31.4978,torch.ones_like(v),dss)
    td=.6+1/(torch.exp(-.05*(v+6))+torch.exp(.09*(v+14)))
    dd=(dss-x[...,23])/td
    fss=1/(1+torch.exp((v+19.58)/3.696))
    tff=7+1/(.0045*torch.exp(-(v+20)/10)+.0045*torch.exp((v+20)/10))
    tfs=1000+1/(.000035*torch.exp(-(v+5)/4)+.000035*torch.exp((v+5)/6))
    dff=(fss-x[...,24])/tff; dfs=(fss-x[...,25])/tfs
    f=.6*x[...,24]+.4*x[...,25]
    tfcaf=7+1/(.04*torch.exp(-(v-4)/7)+.04*torch.exp((v-4)/7))
    tfcas=100+1/(.00012*torch.exp(-v/3)+.00012*torch.exp(v/7))
    Afcaf=.3+.6/(1+torch.exp((v-10)/10)); Afcas=1-Afcaf
    dfcaf=(fss-x[...,26])/tfcaf; dfcas=(fss-x[...,27])/tfcas
    fca=Afcaf*x[...,26]+Afcas*x[...,27]
    jcass=1/(1+torch.exp((v+18.08)/2.7916)); djca=(jcass-x[...,28])/75
    dffp=(fss-x[...,31])/ (2.5*tff)
    dfcafp=(fss-x[...,32])/(2.5*tfcaf)
    fcap=Afcaf*x[...,31]+Afcas*x[...,27]; fp=.6*x[...,31]+.4*x[...,25]
    # Davies activity and ICaL/ICaNa/ICaK driving forces.
    cli,clo=24.,150.; Io=.5*(nao+ko+clo+4*cao)/1000.; Ii=.5*(nass+kss+cli+4*cass)/1000.; constA=1.82e6*(74*310)**(-1.5)
    def gam(I,z):
        I=torch.as_tensor(I,dtype=x.dtype,device=x.device); return torch.exp(-constA*z*(torch.sqrt(I)/(1+torch.sqrt(I))-.3*I))
    gcai,gcao,gnai,gnao,gki,gkao=gam(Ii,4),gam(Io,4),gam(Ii,1),gam(Io,1),gam(Ii,1),gam(Io,1)
    PhiCaLss=4*vffrt*(gcai*cass*torch.exp(2*vfrt)-gcao*cao)/(torch.exp(2*vfrt)-1); PhiCaNass=vffrt*(gnai*nass*torch.exp(vfrt)-gnao*nao)/(torch.exp(vfrt)-1); PhiCaKss=vffrt*(gki*kss*torch.exp(vfrt)-gkao*ko)/(torch.exp(vfrt)-1)
    Io=.5*(nao+ko+clo+4*cao)/1000.; Ii=.5*(nai+ki+cli+4*cai)/1000.; gcai,gcao,gnai,gnao,gki,gkao=gam(Ii,4),gam(Io,4),gam(Ii,1),gam(Io,1),gam(Ii,1),gam(Io,1)
    PhiCaLi=4*vffrt*(gcai*cai*torch.exp(2*vfrt)-gcao*cao)/(torch.exp(2*vfrt)-1); PhiCaNai=vffrt*(gnai*nai*torch.exp(vfrt)-gnao*nao)/(torch.exp(vfrt)-1); PhiCaKi=vffrt*(gki*ki*torch.exp(vfrt)-gkao*ko)/(torch.exp(vfrt)-1)
    PCa=8.3757e-5; PCap=1.1*PCa; PCaNa=.00125*PCa; PCaK=3.574e-4*PCa; PCaNap=.00125*PCap; PCaKp=3.574e-4*PCap
    g1=dgate*(f*(1-nca)+jca*fca*nca); g2=dgate*(fp*(1-nca)+jca*fcap*nca); g3=dgate*(f*(1-nca_i)+jca*fca*nca_i); g4=dgate*(fp*(1-nca_i)+jca*fcap*nca_i)
    fICaLp=fINap
    ICaLss=.8*PhiCaLss*((1-fICaLp)*PCa*g1+fICaLp*PCap*g2)
    ICaNa_ss=.8*PhiCaNass*((1-fICaLp)*PCaNa*g1+fICaLp*PCaNap*g2)
    ICaK_ss=.8*PhiCaKss*((1-fICaLp)*PCaK*g1+fICaLp*PCaKp*g2)
    ICaLi=.2*PhiCaLi*((1-fICaLp)*PCa*g3+fICaLp*PCap*g4)
    ICaNai=.2*PhiCaNai*((1-fICaLp)*PCaNa*g3+fICaLp*PCaNap*g4)
    ICaKi=.2*PhiCaKi*((1-fICaLp)*PCaK*g3+fICaLp*PCaKp*g4)
    ICaL=ICaLss+ICaLi; ICaNa=ICaNa_ss+ICaNai; ICaK=ICaK_ss+ICaKi
    # IKr five-state Markov transition rates and IKs/IK1 currents.
    ikr_c0,ikr_c1,ikr_c2,ikr_o,ikr_i=[x[...,i] for i in range(37,42)]
    vf=v*F/(R*T)
    alpha=.1161*torch.exp(.2990*vf); beta=.2442*torch.exp(-1.604*vf)
    alpha1=torch.full_like(v,.154375); beta1=torch.full_like(v,.1911)
    alpha2=.0578*torch.exp(.9710*vf); beta2=.349e-3*torch.exp(-1.062*vf)
    alphai=.2533*torch.exp(.5953*vf); betai=1.25*.0522*torch.exp(-.8209*vf)
    alphac2ToI=.52e-4*torch.exp(1.525*vf); betaItoC2=(beta2*betai*alphac2ToI)/(alpha2*alphai)
    dc0=ikr_c1*beta-ikr_c0*alpha
    dc1=ikr_c0*alpha+ikr_c2*beta1-ikr_c1*(beta+alpha1)
    dc2=ikr_c1*alpha1+ikr_o*beta2+ikr_i*betaItoC2-ikr_c2*(beta1+alpha2+alphac2ToI)
    do=ikr_c2*alpha2+ikr_i*betai-ikr_o*(beta2+alphai)
    di=ikr_c2*alphac2ToI+ikr_o*alphai-ikr_i*(betaItoC2+betai)
    IKr=.0321*torch.sqrt(torch.as_tensor(ko/5,dtype=x.dtype,device=x.device))*ikr_o*(v-EK)
    xs1,xs2=x[...,33],x[...,34]
    xs1ss=1/(1+torch.exp(-(v+11.60)/8.932)); txs1=817.3+1/(2.326e-4*torch.exp((v+48.28)/17.80)+.001292*torch.exp(-(v+210)/230))
    dxs1=(xs1ss-xs1)/txs1; xs2ss=xs1ss; txs2=1/(.01*torch.exp((v-50)/20)+.0193*torch.exp(-(v+66.54)/31)); dxs2=(xs2ss-xs2)/txs2
    KsCa=1+.6/(1+(3.8e-5/torch.clamp(cai,min=1e-9))**1.4); IKs=.0034*KsCa*xs1*xs2*(v-EKs)
    aK1=4.094/(1+torch.exp(.1217*(v-EK-49.934))); bK1=(15.72*torch.exp(.0674*(v-EK-3.257))+torch.exp(.0618*(v-EK-594.31)))/(1+torch.exp(-.1629*(v-EK+14.207)))
    K1ss=aK1/(aK1+bK1); IK1=.6992*torch.sqrt(torch.as_tensor(ko/5,dtype=x.dtype,device=x.device))*K1ss*(v-EK)
    # Na/Ca exchanger. This is written explicitly to preserve the native
    # algebra and remains differentiable for positive concentrations.
    def ncx(na_i, ca_i, scale_ss):
        kna1,kna2,kna3,kasymm,wna,wca,wnaca,kcaon,kcaoff=15.,5.,88.12,12.5,6e4,6e4,5e3,1.5e6,5e3
        qna,qca=.5224,.1670; hca=torch.exp(qca*v*F/(R*T)); hna=torch.exp(qna*v*F/(R*T))
        h1=1+na_i/kna3*(1+hna); h2=na_i*hna/(kna3*h1); h3=1/h1
        h4=1+na_i/kna1*(1+na_i/kna2); h5=na_i*na_i/(h4*kna1*kna2); h6=1/h4
        h7=1+nao/kna3*(1+1/hna); h8=nao/(kna3*hna*h7); h9=1/h7
        h10=kasymm+1+nao/kna1*(1+nao/kna2); h11=nao*nao/(h10*kna1*kna2); h12=1/h10
        k1=h12*cao*kcaon; k2=kcaoff; k3p=h9*wca; k3pp=h8*wnaca; k3=k3p+k3pp
        k4p=h3*wca/hca; k4pp=h2*wnaca; k4=k4p+k4pp; k5=kcaoff; k6=h6*ca_i*kcaon; k7=h5*h2*wna; k8=h8*h11*wna
        q1=k2*k4*(k7+k6)+k5*k7*(k2+k3); q2=k1*k7*(k4+k5)+k4*k6*(k1+k8)
        q3=k1*k3*(k7+k6)+k8*k6*(k2+k3); q4=k2*k8*(k4+k5)+k3*k5*(k1+k8); den=q1+q2+q3+q4
        E1,E2,E3,E4=q1/den,q2/den,q3/den,q4/den
        allo=1/(1+(150e-6/torch.clamp(ca_i,min=1e-9))**2)
        jna=3*(E4*k7-E1*k8)+E3*k4pp-E2*k3pp; jca=E2*k2-E1*k1
        return scale_ss*.0034*allo*(jna+2*jca)
    INaCa_i=ncx(nai,cai,.65); INaCa_ss=ncx(nass,cass,.2)
    # Na/K pump and background currents.
    Knai0,Knao0,delta=9.073,27.78,-.155; Knai=Knai0*torch.exp(delta*v*F/(3*R*T)); Knao=Knao0*torch.exp((1-delta)*v*F/(3*R*T))
    Kki,Kko=.5,.3582; MgADP,MgATP=.05,9.8; Kmgatp, H,Khp,Knap,Kxkur=1.698e-7,1e-7,1.698e-7,224.,292.; eP=4.2
    Ppump=eP/(1+H/Khp+nai/Knap+ki/Kxkur)
    k1p,k1m,k2p,k2m,k3p,k3m,k4p,k4m=949.5,182.4,687.2,39.4,1899.,79300.,639.,40.
    a1=(k1p*(nai/Knai)**3)/((1+nai/Knai)**3+(1+ki/Kki)**2-1); b1=k1m*MgADP; a2=k2p
    b2=(k2m*(nao/Knao)**3)/((1+nao/Knao)**3+(1+ko/Kko)**2-1); a3=(k3p*(ko/Kko)**2)/((1+nao/Knao)**3+(1+ko/Kko)**2-1)
    b3=(k3m*Ppump*H)/(1+MgATP/Kmgatp); a4=(k4p*MgATP/Kmgatp)/(1+MgATP/Kmgatp); b4=(k4m*(ki/Kki)**2)/((1+nai/Knai)**3+(1+ki/Kki)**2-1)
    q1=a4*a1*a2+b2*b4*b3+a2*b4*b3+b3*a1*a2; q2=b2*b1*b4+a1*a2*a3+a3*b1*b4+a2*a3*b4
    q3=a2*a3*a4+b3*b2*b1+b2*b1*a4+a3*a4*b1; q4=b4*b3*b2+a3*a4*a1+b2*a4*a1+b3*b2*a1; den=q1+q2+q3+q4
    E1,E2,E3,E4=q1/den,q2/den,q3/den,q4/den; INaK=30*(3*(E1*a3-E2*b3)+2*(E4*b1-E3*a1))
    xkb=1/(1+torch.exp(-(v-10.8968)/23.9871)); IKb=.0189*xkb*(v-EK)*.9
    INab=1.9239e-9*vffrt*(nai*torch.exp(vfrt)-nao)/(torch.exp(vfrt)-1)
    ICab=5.9194e-8*4*vffrt*(gcai*cai*torch.exp(2*vfrt)-gcao*cao)/(torch.exp(2*vfrt)-1)
    IpCa=5e-4*cai/(.0005+cai); ICap=(1e-6*10.08*stretch_d)*4*vffrt*(gcai*cai*torch.exp(2*vfrt)-gcao*cao)/(torch.exp(2*vfrt)-1)
    ecl=(R*T/F)*torch.log(torch.as_tensor(24/150,dtype=x.dtype,device=x.device)); IClCa=.2843/(1+.1/cass)*(v-ecl); IClbk=.00198*(v-ecl)
    # Calcium release, uptake, and diffusion fluxes.
    fJrelp=1/(1+KmCaMK/CaMKa); bt=4.75; Jrel_inf=.5*bt*(-ICaL)/(1+(1.7/torch.clamp(cajsr,min=1e-9))**8); tau_rel=bt/(1+.0123/torch.clamp(cajsr,min=1e-9)); tau_rel=torch.clamp(tau_rel,min=.001); dJrelnp=(Jrel_inf-x[...,35])/tau_rel
    btp=1.25*bt; Jrel_infp=.5*btp*(-ICaL)/(1+(1.7/torch.clamp(cajsr,min=1e-9))**8); tau_relp=torch.clamp(btp/(1+.0123/torch.clamp(cajsr,min=1e-9)),min=.001); dJrelp=(Jrel_infp-x[...,42])/tau_relp
    Jrel=1.5378*((1-fJrelp)*x[...,35]+fJrelp*x[...,42]); fJupp=1/(1+KmCaMK/CaMKa)
    Jupnp=.005425*cai/(cai+.00092); Jupp=2.75*.005425*cai/(cai+.00092-.00017); Jleak=.0048825*cansr/15; Jup=(1-fJupp)*Jupnp+fJupp*Jupp-Jleak; Jtr=(cansr-cajsr)/60
    JdiffNa=(nass-nai)/2; JdiffK=(kss-ki)/2; Jdiff=(cass-cai)/.2; Kmn=.002; k2n=500.; km2n=jca; anca=1/(k2n/km2n+(1+Kmn/cass)**4); dnca=anca*k2n-nca*km2n; anca_i=1/(k2n/km2n+(1+Kmn/cai)**4); dnca_i=anca_i*k2n-nca_i*km2n
    # State derivatives (celltype=0, fixed constants of model_SAC).
    vcell=1000*3.141592653589793*(.0011**2)*.01; vmyo=.68*vcell; vnsr=.0552*vcell; vjsr=.0048*vcell; vss=.02*vcell; Acap=2*(2*3.141592653589793*(.0011**2)+2*3.141592653589793*.0011*.01)
    ISt=torch.where(torch.as_tensor(t,dtype=x.dtype,device=x.device)<=1,
                    torch.full_like(v,-53.),torch.zeros_like(v))
    Bcai=1/(1+.05*.00238/(.00238+cai)**2+.07*.0005/(.0005+cai)**2); Bcass=1/(1+.047*.00087/(.00087+cass)**2+1.124*.0087/(.0087+cass)**2); Bcajsr=1/(1+10*.8/(.8+cajsr)**2)
    dv=-(INa+INaL+Ito+ICaL+ICaNa+ICaK+IKr+IKs+IK1+INaCa_i+INaCa_ss+INaK+INab+IKb+IpCa+ICab+IClCa+IClbk+ISt+INs+IKo+ICap)
    dnai=-(INa+INaL+3*INaCa_i+3*INaK+INab+INsNa)*Acap/(F*vmyo)+JdiffNa*vss/vmyo; dnass=-(ICaNa+3*INaCa_ss)*Acap/(F*vss)-JdiffNa
    dki=-(ICaKi+Ito+IKr+IKs+IK1+IKb+ISt-2*INaK+INsK+IKo)*Acap/(F*vmyo)+JdiffK*vss/vmyo; dkss=-ICaK_ss*Acap/(F*vss)-JdiffK
    dcai=Bcai*(-(ICaLi+IpCa+ICab-2*INaCa_i+ICap)*Acap/(2*F*vmyo)-Jup*vnsr/vmyo+Jdiff*vss/vmyo); dcass=Bcass*(-(ICaLss-2*INaCa_ss)*Acap/(2*F*vss)+Jrel*vjsr/vss-Jdiff); dcansr=Jup-Jtr*vjsr/vnsr; dcajsr=Bcajsr*(Jtr-Jrel)
    deriv=torch.stack([dv,dnai,dnass,dki,dkss,dcai,dcass,dcansr,dcajsr,dm,dhp,dh,dj,djp,dmL,dhL,dhLp,da,diF,diS,dap,diFp,diSp,dd,dff,dfs,dfcaf,dfcas,djca,dnca,dnca_i,dffp,dfcafp,dxs1,dxs2,dJrelnp,dCaMKt,dc0,dc1,dc2,do,di,dJrelp],dim=-1)
    if return_intermediates:
        return locals()
    return deriv


















