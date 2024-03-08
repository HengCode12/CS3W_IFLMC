function [result_2way,result_3way]  =  result_CSCI2WD(test_label_c, prob, cost_matrix_2decision,cost_matrix_3decision, Label_Decision)
% for multi-classfication
% test_label_c:  No. of rows: test_num  列向量   正类为1，负类为2
% prob: the classification probabilities, each row is for one sample
%       every column is the probability for one class
% cost_matrix: the misclassification cost matrix, size: n*n
%              n is the number of categories. class_num

% result_2way: the decision, decision cost and error_rate for both methods
%------------------------------------------------------------

% test_label_c = test_label';
% prob = [loop.prob{1,1}]';
%cost_matrix = 2*(ones(M,M)-eye(M,M));
%test------------------------------------

class_num = size(prob,2); % number of categories
    assert(class_num==size(cost_matrix_2decision,1),'size of prob and cost_matrix?')
% assert(a_trust>0,'alpha > 0 ?')
test_num = size(test_label_c,1);  % number of test samples
    assert(test_num == size(prob,1),'test_label prob ?')
eachclass_num = test_num / class_num;%每一类样本数

%%CS3WD
bayes_loss_3WD = prob*cost_matrix_3decision;% the expected loss of test samples, bayes loss
[loss_sort, num_sort]=sort(bayes_loss_3WD,2,'ascend');%代价最小的决策
dec_CS3WD = num_sort(:,1);% size: n*1
    assert(test_num == size(dec_CS3WD,1),'decision CS3WD ?')

dec_res1_CS3WD = dec_CS3WD == test_label_c;  % 正确判断的结果 
dec_res2_CS3WD = dec_CS3WD==3;       %判断为边界
bound_num = length(find(dec_res2_CS3WD==1));
bound_index = find(dec_res2_CS3WD==1);
non_bound_num = length(dec_res2_CS3WD) - bound_num;
TP_CS3WD = length(find(dec_res1_CS3WD(find(test_label_c==1))==1));
TN_CS3WD = length(find(dec_res1_CS3WD(find(test_label_c==2))==1));
FN_CS3WD = length(find(dec_CS3WD(find(test_label_c==1))==2));
FP_CS3WD = length(find(dec_CS3WD(find(test_label_c==2))==1));
acc_CS3WD =  (TP_CS3WD+TN_CS3WD)/(TP_CS3WD+TN_CS3WD+FN_CS3WD+FP_CS3WD); 
pre_CS3WD = TP_CS3WD/(TP_CS3WD + FP_CS3WD);
recall_CS3WD = TP_CS3WD/(TP_CS3WD + FN_CS3WD);
F1_CS3WD = 2*pre_CS3WD*recall_CS3WD/(pre_CS3WD+recall_CS3WD);

cost_i_CS3WD = zeros(size(dec_CS3WD));
for i=1:test_num
    cost_i_CS3WD(i) = cost_matrix_3decision(test_label_c(i),dec_CS3WD(i));
end
cost_CS3WD = sum(cost_i_CS3WD);%总代价

cost_i_orgin = zeros(size(dec_CS3WD));

num = find(Label_Decision == -1);
Label_Decision(num) = 2;
for i=1:test_num
    cost_i_orgin(i) = cost_matrix_2decision(test_label_c(i),Label_Decision(i));
end
cost_orgin = sum(cost_i_orgin);%总代价

test_label_c_2 = test_label_c;
test_label_c_2(bound_index) = [];





%% CS2WD代价敏感二支决策
bayes_loss = prob*cost_matrix_2decision;% the expected loss of test samples, bayes loss
[loss_sort, num_sort]=sort(bayes_loss,2,'ascend');  %按行，升序
dec_CS2WD = num_sort(:,1);   %贝叶斯损失最小做出CS2WD的决策
assert(test_num == size(dec_CS2WD,1),'decision CS2WD ?')   %检查是否有错
dec_CS2WD_2 = dec_CS2WD;
dec_CS2WD_2(bound_index) = [];
dec_res_CS = dec_CS2WD_2 == test_label_c_2;   %决策与真实相等为1，否则为0
TP_CS2WD = length(find(dec_res_CS(find(test_label_c_2==1))==1));
TN_CS2WD = length(find(dec_res_CS(find(test_label_c_2==2))==1));
FN_CS2WD = length(find(dec_res_CS(find(test_label_c_2==1))==0));
FP_CS2WD = length(find(dec_res_CS(find(test_label_c_2==2))==0));
acc_CS2WD = (TP_CS2WD+TN_CS2WD)/(TP_CS2WD+TN_CS2WD+FN_CS2WD+FP_CS2WD);
pre_CS2WD = TP_CS2WD/(TP_CS2WD + FP_CS2WD);
recall_CS2WD = TP_CS2WD/(TP_CS2WD + FN_CS2WD);
F1_CS2WD = 2*pre_CS2WD*recall_CS2WD/(pre_CS2WD+recall_CS2WD);


cost_i_CS = zeros(size(dec_CS2WD));
for i=1:test_num
    cost_i_CS(i) = cost_matrix_2decision(test_label_c(i),dec_CS2WD(i));
end
cost_CS2WD = sum(cost_i_CS);%总代价

% sum_rate = 0;
% for i=1:class_num
%     class_r = sum(dec_CS3WD((eachclass_num*(i-1)+1):(eachclass_num*i),1).*pos_reg((eachclass_num*(i-1)+1):(eachclass_num*i)) ...
%         == test_label_c((eachclass_num*(i-1)+1):(eachclass_num*i)));
%     class_pos = sum(dec_CS2WD((eachclass_num*(i-1)+1):(eachclass_num*i)));
%     class_rate = class_r / class_pos;
%     sum_rate = sum_rate + class_rate;
% end
% ave_rate = sum_rate / class_num;%CCA / CAR





%% CI2WD
[prob_sort,cate_sort]=sort(prob,2,'descend'); 
dec_CI2WD = cate_sort(:,1);
assert(test_num == size(dec_CI2WD,1),'decision CI2WD ?')
dec_CI2WD_2 = dec_CI2WD;
dec_CI2WD_2(bound_index) = [];
dec_res_CI = dec_CI2WD_2 == test_label_c_2;%概率最大的决策
TP_CI2WD = length(find(dec_res_CI(find(test_label_c_2==1))==1));
TN_CI2WD = length(find(dec_res_CI(find(test_label_c_2==2))==1));
FN_CI2WD = length(find(dec_res_CI(find(test_label_c_2==1))==0));
FP_CI2WD = length(find(dec_res_CI(find(test_label_c_2==2))==0));
acc_CI2WD =  (TP_CI2WD+TN_CI2WD)/(TP_CI2WD+TN_CI2WD+FN_CI2WD+FP_CI2WD);
pre_CI2WD = TP_CI2WD/(TP_CI2WD + FP_CI2WD);
recall_CI2WD = TP_CI2WD/(TP_CI2WD + FN_CI2WD);
F1_CI2WD = 2*pre_CI2WD*recall_CI2WD/(pre_CI2WD+recall_CI2WD);
% right_P2_num = sum( cate_sort(:,2)==test_label_c)
% the second most possible choice

cost_i_CI = zeros(size(test_label_c));
for i=1:test_num
    cost_i_CI(i) = cost_matrix_2decision(test_label_c(i),dec_CI2WD(i));
end
cost_CI2WD = sum(cost_i_CI);%总代价






%% output
result_2way.dec = [dec_CS2WD dec_CI2WD];
result_2way.cost = [cost_CS2WD cost_CI2WD];
result_2way.acc = [acc_CS2WD acc_CI2WD];
result_2way.pre = [pre_CS2WD pre_CI2WD];
result_2way.recall = [recall_CS2WD recall_CI2WD];
result_2way.F1 = [F1_CS2WD F1_CI2WD];
result_2way.bayes_risk = bayes_loss;
result_2way.CS_detail = [TP_CS2WD TN_CS2WD FN_CS2WD FP_CS2WD]
result_2way.CI_detail = [TP_CI2WD TN_CI2WD FN_CI2WD FP_CI2WD]
result_2way.ord = 'CS2WD CI2WD'

result_3way.dec = dec_CS3WD;
result_3way.cost = cost_CS3WD;
result_3way.acc = acc_CS3WD;
result_3way.pre = pre_CS3WD;
result_3way.recall = recall_CS3WD;
result_3way.F1 = F1_CS3WD;
result_3way.bayes_risk = bayes_loss_3WD;
result_3way.ord = 'CS3WD'
result_3way.detail = [bound_num non_bound_num TP_CS3WD TN_CS3WD FN_CS3WD FP_CS3WD]
result_3way.cost_orgin = cost_orgin;
result_3way.bound_index = bound_index;
