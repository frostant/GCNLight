#include <iostream>
#include <cstring>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdio>
#define N 10000005
#define rep(i,a,b) for(int i=a;i<=b;i++)
using namespace std;
int a[N],b[N],n,m; 
int tot=1,to[N],head[N],net[N],edge,w[N],fl[N],q[N],ans[N],ss[N]; 
int pro=0.67,epoch=100;

void add(int frm,int to2){
    to[++tot]=to2, net[tot]=head[frm], head[frm]=tot, w[tot]++;
    to[++tot]=frm, net[tot]=head[to2], head[to2]=tot, w[tot]++;
}
void clear(int x,int v){
    w[(x)*2]+=v, w[(x*2)^1]+=v;
}
int dfs(int frm,int tos){
    int x,l=0,r=1; 
    rep(i,1,n+m) fl[i]=0; 
    q[0]=frm; fl[frm]=1; 
    while(r>l){
        x=q[l++];
        for(int t=head[x];t;t=net[t]){
            if(!fl[to[t]] && w[t]){
                fl[to[t]]=fl[x]+1;
                q[r++]=to[t]; 
                if(to[t]==tos){
                    return fl[to[t]]; 
                }
            }
        }
    }
    return -1; 
}
int main (){
    // freopen("../data/small.txt","r",stdin);
    // freopen("../data/gowalla.train","r",stdin);
    freopen("../data/lastfm.train","r",stdin);
    char ch;
    int aa,bb;
    // scanf("%d%c%d",&aa,&ch,&bb);
    // a[++edge]=aa, b[edge]=bb;
    // scanf("%d%c%d",&aa,&ch,&bb);
    // a[++edge]=aa, b[edge]=bb;
    // scanf("%d%c%d",&aa,&ch,&bb);
    // a[++edge]=aa, b[edge]=bb;
    // scanf("%d%c%d",&a[++edge],&ch,&b[edge]);
    // scanf("%d%c%d",&a[++edge],&ch,&b[edge]);
    // cout<<a[2]<<" "<<b[2]<<" "<<ch<<endl;

    // return 0; 
    while(scanf("%d%c%d",&aa,&ch,&bb)!=EOF){
    // while(scanf("%d%d",&aa,&bb)!=EOF){
        a[++edge]=aa+1, b[edge]=bb+1; 
        // cout<<a[edge]<<" "<<b[edge]<<endl; 
        n=max(n,a[edge]); 
        m=max(m,b[edge]); 
         
    }
    printf("Node:%d, %d Edge:%d\n",n,m,edge);
    rep(i,1,edge){
        add(a[i],b[i]+n); 
    }
    int tt=0,ma=0,non=0; 
    rep(i,1,edge){
        if (i%5000==0) cout<<"now:"<<i<<endl;
        clear(i,-1);
        // cout<<a[i]<<" "<<b[i]<<" "<<w[i*2]<<" "<<w[i*2+1]<<endl;
        tt=dfs(a[i],b[i]+n); 
        clear(i,1);  
        ans[i]=tt; 
        if (tt==-1){
            non++; 
        }
        else ss[tt]++;
        ma=max(ma,tt); 
    }
    rep(i,1,edge){
        // if(ans[i]/)
        printf("%d:edge %d\n",i,ans[i]); 
    }
    rep(i,0,ma) if(ss[i]) printf("%d:ans %d\n",i,ss[i]); 
    printf("-1:%d\n",non); 
    return 0;
}
// lastfm
// 4:ans 40405
// 6:ans 1594
// 8:ans 4
// -1:132

// gowalla
// Node:29858, 40981 Edge:810128
// 4:ans 803695
// 6:ans 6425
// -1:8

// 数组太小
// 2:ans 185390
// 3:ans 55310
// 4:ans 387420
// 5:ans 178794
// 6:ans 110561
// 7:ans 49928
// 8:ans 9602
// 9:ans 1287
// 10:ans 140
// 11:ans 7
// -1:13976
// g++ -o a.exe dataAugment.cpp && ./a.exe