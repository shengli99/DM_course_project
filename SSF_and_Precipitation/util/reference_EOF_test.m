%���ƽ
M=reValue;       % or M=dlmread('RAIN_mask.dat');
 M=dlmread('RAIN_mask_13mean.dat');
X=detrend(M,0)';

%EOFʱ��ת�����������������Ϻ�Ԥ�ⷽ������鱦 P28
C=X'*X/valuenum;   %Э�������C
[EOF0,E]=eig(C); %Э�������C������ֵE����������EOF0  

E=rot90(E,2); %���Խ�������ֵ�Ӵ�С����
EOF0=fliplr(EOF0);%�������������ҵߵ�

EOF=X*EOF0;   %Ŀ��Э�������XX'/n����������
E=E*valuenum/n; %Ŀ��Э�������XX'/n������ֵ���Ӵ�С����
lambda=diag(E);%������ֵ��ȡ����
clear C EOF0 E 
sq=[sqrt(lambda*n)+eps]';   % 1*n 
sq=sq(ones(1,valuenum),:); %  valnum * n
EOF=EOF./sq;  % �ռ亯�� valnum*n ��i�д����iģ̬�Ŀռ亯�� 
PC=EOF'*X;    % ʱ�亯�� n*n ��i�д����iģ̬��ʱ��ϵ��
clear sq; 
%�����ɷֺ�����������׼��
EOF=EOF.*sqrt(repmat(lambda',valuenum,1));
PC=PC./sqrt(repmat(lambda,1,n));
dlmwrite('RAIN_PC.dat',PC,'delimiter',' ','newline','pc'); %
dlmwrite('RAIN_EOF.dat',EOF,'delimiter',' ','newline','pc'); %


%EOF�������Լ��顪��North��1982�� 
n_eof=0;
for i=1:n
    err=lambda(i)*sqrt(2/n);
    if(lambda(i)-lambda(i+1)>=err)
        continue;
    else
        n_eof=i;   %�˳���n_eof=7��˵��ǰ�߸�ģ̬������Ч��
        break;
    end
end
clear err i 

%�ۻ��������ͼ 
cum=100*cumsum(lambda)./sum(lambda);%�ۻ��������ͼ
percent_explained=100*lambda./sum(lambda);%�����
%----EOF(monthly anomaly):ǰ�߸�ģ̬��Ч��ǰ����ģ̬ռ��20.8%,7.4%,5.1%����������%
%----EOF(13�㻬��ƽ��):ǰ����ģ̬ռ��36.54%,8.5%,6.3%��-------------------------%
figure
pareto(percent_explained)
xlabel('���ɷ�','fontsize',12,'fontname','����')
ylabel('�������(%)','fontsize',12,'fontname','����')
set(gcf,'color','w');
dlmwrite('PerctExpln_EOF.dat',percent_explained,'delimiter',' ','newline','pc');
dlmwrite('PerctCum_EOF.dat',cum,'delimiter',' ','newline','pc');
