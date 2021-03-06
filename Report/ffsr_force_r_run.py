
import rpy2.robjects as ro
import pandas.rpy.common as com

ro.r("""fsr.fast.include<-function(x,y,gam0=.05,digits=4,print=T,inc){
# estimated alpha for forward selection
# this program allows variables to be forced in
# for example inc=c(12,3,5) forces in variables in columns 12,3, and 5 of x
# not set up to handle inc=NULL, use fsr.fast when not including variables
require(leaps)
ok<-complete.cases(x,y)
x<-x[ok,]                            # get rid of na's
y<-y[ok]                             # since regsubsets can't handle na's
m<-ncol(x)
n<-nrow(x)
colnames(x)<- as.character(1:m)
m.inc=length(inc)
inc.reo=c(inc,setdiff(1:m,inc))        # new order for x's, inc at beginning
if(m >= n) m1 <- n-5  else m1<-m     # to get rid of NA's in pv
vm<-1:m1
as.matrix(x)->x                      # in case x is a data frame
pvm<-rep(0,m1)                       # to create pvm below
regsubsets(x,y,force.in=inc,method="forward")->out.x
ch=out.x$vorder-1
vorder.new=inc.reo[ch[2:(m1+1)]]     # order without intercept
pv.orig<-1-pf((out.x$rss[vm]-out.x$rss[vm+1])*(out.x$nn-(vm+1))/out.x$rss[vm+1],1,out.x$nn-(vm+1))
pv.orig[1:m.inc]=rep(0,m.inc)
for (i in (m.inc+1):m1){pvm[i]<-max(pv.orig[1:i])}  # sequential max of pvalues
alpha<-c(0,pvm)
ng<-length(alpha)
S<-rep(0,ng)                         # will contain num. of true entering in orig.
for (ia in 2:ng){                    # loop through alpha values for S=size
S[ia] <- sum(pvm<=alpha[ia])         # size of models at alpha[ia], S[1]=0
}
alphas<-gam0 * (1. + S[2:ng]) / (m - S[2:ng])
ghat<-(m-S)*alpha/(1+S)              # gammahat_ER
real.seq<-data.frame(S=S[2:ng],var=vorder.new,pval=pv.orig,
         pvmax=pvm,alpha=alphas,g=ghat[2:ng])#,Rsq=round(1-out.x$rss[2:(m1+1)]/out.x$rss[1],4))
alpha<-c(0,pvm[(m.inc+1):m1])        # note alpha reduced by number forced in
ng<-length(alpha)
S<-rep(0,ng)                         # will contain num. of true entering in orig.
pvm=pvm[(m.inc+1):m1]                # redefine to get rid of 0's at beginnning
for (ia in 2:ng){                    # loop through alpha values for S=size
S[ia] <- sum(pvm<=alpha[ia])         # size of models at alpha[ia], S[1]=0
}
ghat<-(m-m.inc-S)*alpha/(1+S)              # gammahat_ER
####
if(print)print(round(real.seq,digits),S)
# add additional points to make jumps
alpha2<-alpha[2:ng]-.0000001
ghat2<-(m-m.inc-S[1:(ng-1)])*alpha2/(1+S[1:(ng-1)])
zp<-data.frame(a=c(alpha,alpha2),g=c(ghat,ghat2))
zp<-zp[order(zp$a),]
gmax<-max(zp$g)
index.max<-which.max(zp$g)           # index of largest ghat
alphamax<-zp$a[index.max]            # alpha with largest ghat
ind<-(ghat <= gam0 & alpha<=alphamax)*1
Sind<-S[max(which(ind > 0))]          # model size with ghat just below gam0
alphahat.fast<-(1+Sind)*gam0/(m-m.inc-Sind)  # ER est.
size<-sum(pvm<=alphahat.fast)+m.inc       # size of model without intercept
colnames(x)<-colnames(x,do.NULL=F,prefix="")      # corrects for no colnames
x<-x[,colnames(x)[vorder.new[1:size]]]
x.ind<-vorder.new[1:size]
mod <- lm(y~x)
return(list(mod=mod,size=size,x.ind=x.ind,alphahat.ER=alphahat.fast,inc=inc))
}""")

ro.r('ncaa <- as.matrix(read.table("ncaa_data2.txt",header=T))')

print "NCSU R Results:\n"

print ro.r('system.time(fsr.fast.include(x=ncaa[,1:19],y=ncaa[,20],inc=c(12,3,5)))')