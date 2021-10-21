
library(tidyverse)
library(gridExtra)

# library(MLmetrics)
library(ModelMetrics)

library(caret)

library(reshape2)
library(pROC)

library(effsize)
library(ScottKnottESD)

save.fig.dir = '../output/figure/'

dir.create(file.path(save.fig.dir), showWarnings = FALSE)

preprocess <- function(x, reverse){
  colnames(x) <- c("variable","value")
  tmp <- do.call(cbind, split(x, x$variable))
  tmp <- tmp[, grep("value", names(tmp))]
  names(tmp) <- gsub(".value", "", names(tmp))
  df <- tmp
  ranking <- NULL
  
  if(reverse == TRUE)
  { 
    ranking <- (max(sk_esd(df)$group)-sk_esd(df)$group) +1 
  }
  else
  { 
    ranking <- sk_esd(df)$group 
  }
  
  x$rank <- paste("Rank",ranking[as.character(x$variable)])
  return(x)
}

# ---------------- Code for RQ1 and RQ3 (by AJ. Pick) -----------------------#

# current one
# prediction_dir = '../output/prediction/DeepLineDP/rebalancing-adaptive-ratio2-lowercase-with-comment-50-dim-6-epochs/'
# prediction_dir = '../output/prediction/DeepLineDP/no-rebalancing-adaptive-ratio2-lowercase-with-comment-50-dim-14-epochs/'

prediction_dir = '../output/prediction/DeepLineDP/rebalancing-adaptive-ratio2-lowercase-with-comment-small-batch-50-dim-10-epochs/'


all_files = list.files(prediction_dir)

java_keywords = read.csv("java_keywords.csv",header = F)

df_all <- NULL

for(f in all_files)
{
  df <- read.csv(paste0(prediction_dir, f))
  df_all <- rbind(df_all, df)
}

#RQ1-1
df_all %>% filter(is.comment.line=="False" & file.level.ground.truth=="True" & prediction.label=="True") %>% group_by(test, filename,token) %>%
  summarise(range=max(token.attention.score)-min(token.attention.score), sd=sd(token.attention.score)) %>%
  melt() %>% ggplot(aes(x=variable, y=value)) + geom_boxplot() + scale_y_continuous(breaks=0:4*0.25)


#RQ1-2
# for token.attention
# common_tokens = df_all %>% group_by(test, filename, token) %>% summarize(in_defective_line = sum(line.level.ground.truth == "True") > 0, in_clean_line = sum(line.level.ground.truth == "False") > 0 ) %>% filter(in_defective_line == TRUE & in_clean_line == TRUE)
# 
# common_tokens %>% select(test, filename, token) %>% merge(df_all)  %>% ggplot(aes(x=line.level.ground.truth,y=token.attention.score)) + geom_boxplot() +ylab("token score")

df_all_copy = data.frame(df_all)

# df_all_copy[df_all_copy$token %in% java_keywords$V1,]$token.attention.score = 0
# df_all[df_all$token %in% common_tokens$token,]$token.attention.score = 0 # not work

# ---------------------------RQ1-2 new idea----------------------------#

## get max score of each code token in actual buggy lines
## get min score of each code token in actual clean lines
## then plot based on buggy lines and clean lines

df_all_copy = filter(df_all_copy, is.comment.line=="False" & file.level.ground.truth=="True" & prediction.label=="True")

clean.lines.df = filter(df_all_copy, line.level.ground.truth=="False")
buggy.line.df = filter(df_all_copy, line.level.ground.truth=="True")

clean.lines.token.score = clean.lines.df %>% group_by(test, filename, token) %>% summarise(score = min(token.attention.score))
clean.lines.token.score$class = "Clean Lines"

buggy.lines.token.score = buggy.line.df %>% group_by(test, filename, token) %>% summarise(score = max(token.attention.score))
buggy.lines.token.score$class = "Defective Lines"

all.lines.token.score = rbind(clean.lines.token.score, buggy.lines.token.score)

all.lines.token.score %>% ggplot(aes(x=class, y=score)) + geom_boxplot()  

res = cliff.delta(buggy.lines.token.score$score, clean.lines.token.score$score)

# ---------------------------RQ1-2 old idea----------------------------#

get.top.k.tokens = function(df, k)
{
  top.k <- df %>% filter( is.comment.line=="False"  & file.level.ground.truth=="True" & prediction.label=="True" ) %>%
    group_by(test, filename) %>% top_n(k, token.attention.score) %>% select("project","train","test","filename","token") %>% distinct()
  
  top.k$flag = 'topk'
  # tmp.top.k = df %>% merge(top.k, by=c("project","train","test","filename","token"))
  # tmp.top.k = df %>% merge(top.k, by=c("project","train","test","filename","token"), all.x=TRUE)

  return(top.k)
}
# 
# tmp.top10 = get.top.k.tokens(df_all_copy, 10)
# tmp.top50 = get.top.k.tokens(df_all_copy, 50)
# tmp.top100 = get.top.k.tokens(df_all_copy, 100)

# top10 <- df_all_copy %>% filter( is.comment.line=="False"  & file.level.ground.truth=="True" & prediction.label=="True" ) %>%
#   group_by(test, filename) %>% top_n(10, token.attention.score) %>% select("project","train","test","filename","token") %>% distinct()
# tmp.top10 = df_all_copy %>% merge(top10, by=c("project","train","test","filename","token"))
# 
# top50 <- df_all_copy %>% filter( is.comment.line=="False"  & file.level.ground.truth=="True" & prediction.label=="True" ) %>%
#   group_by(test, filename) %>% top_n(50, token.attention.score) %>% select("project","train","test","filename","token") %>% distinct()
# tmp.top50 = df_all_copy %>% merge(top50, by=c("project","train","test","filename","token"))
# 
# top100 <- df_all_copy %>% filter( is.comment.line=="False"  & file.level.ground.truth=="True" & prediction.label=="True" ) %>%
#   group_by(test, filename) %>% top_n(100, token.attention.score) %>% select("project","train","test","filename","token") %>% distinct()
# tmp.top100 = df_all_copy %>% merge(top100, by=c("project","train","test","filename","token"))

# tmp.top10  %>% group_by(test, filename, line.number , line.level.ground.truth)   %>%  summarise(line_score=max(token.attention.score)) %>%
#   ggplot(aes(x=line.level.ground.truth, y=line_score)) + geom_boxplot()  
# 
# tmp.top50  %>% group_by(test, filename, line.number , line.level.ground.truth)   %>%  summarise(line_score=max(token.attention.score)) %>%
#   ggplot(aes(x=line.level.ground.truth, y=line_score)) + geom_boxplot()  
# 
# tmp.top100  %>% group_by(test, filename, line.number , line.level.ground.truth)   %>%  summarise(line_score=max(token.attention.score)) %>%
#   ggplot(aes(x=line.level.ground.truth, y=line_score)) + geom_boxplot()  


# for plotting result of all tokens
# df_all_copy %>% filter( is.comment.line=="False"  & file.level.ground.truth=="True" & prediction.label=="True" ) %>% group_by(test, filename, line.number , line.level.ground.truth)   %>%  summarise(line_score=max(token.attention.score)) %>%
#   ggplot(aes(x=line.level.ground.truth, y=line_score)) + geom_boxplot() 

# for line.attention
# df_all_lines = df_all %>% select(project, train, test, filename, file.level.ground.truth, line.level.ground.truth, prediction.label, is.comment.line, line.attention.score)
# df_all_lines = distinct(df_all_lines)
# 
# df_all_lines = df_all_lines %>% filter(file.level.ground.truth=="True" & prediction.label=="True")
# 
# df_no_comment = df_all_lines %>% filter(is.comment.line=="False"  & file.level.ground.truth=="True" & prediction.label=="True")
# 
# g = ggplot(df_all_lines, aes(x=line.level.ground.truth, y=line.attention.score)) + geom_boxplot() 
# 
# g2 = ggplot(df_no_comment, aes(x=line.level.ground.truth, y=line.attention.score)) + geom_boxplot() 



#RQ1-2
# sorted  %>% ggplot(aes(x=line.level.ground.truth,y=attention_score)) + geom_boxplot() + ylab("sum attention scores")
# #Get Tokens that appear in *both* defective and clean lines
# common_tokens = df_all %>% group_by(test, filename, token) %>% summarize(in_defective_line = sum(line.level.ground.truth == "True") > 0, in_clean_line = sum(line.level.ground.truth == "False") > 0 ) %>% filter(in_defective_line == TRUE & in_clean_line == TRUE)
# 
# common_tokens %>% select(test, filename, token) %>% merge(df_all)  %>% ggplot(aes(x=line.level.ground.truth,y=token.attention.score)) + geom_boxplot() +ylab("token score")
# #distribution of attn of each token
# df_all %>% filter(file.level.ground.truth=="True" & prediction.label=="True") %>% ggplot(aes(x=line.level.ground.truth,y=attention.score)) + geom_boxplot() +ylab("token score")
# #distribution of sum attn of each line
# df_all %>% filter( file.level.ground.truth=="True" & prediction.label=="True") %>% group_by(test, filename, line.number, line.level.ground.truth) %>% summarize(attention_line_score = sum(attention.score)) %>%
#   ggplot(aes(x=line.level.ground.truth,y=attention_line_score)) + geom_boxplot() +ylab("line score (sum)")
# summary(df_all)
# df_all %>% filter(is.na(line.level.ground.truth)) %>% summary()


# ----------------------------Code for RQ2 ---------------------------------#

get.file.level.eval.result = function(prediction.dir, method.name)
{
  all_files = list.files(prediction.dir)

  all.auc = c()
  all.mcc = c()
  all.bal.acc = c()
  all.test.rels = c()

  for(f in all_files) # for looping through files
  {
    df = read.csv(paste0(prediction.dir, f))

    if(method.name == "DeepLineDP")
    {
      df = as_tibble(df)
      df = select(df, c(train, test, filename, file.level.ground.truth, prediction.prob, prediction.label))
      
      df = distinct(df)
    }
    
    all.gt = df$file.level.ground.truth
    all.prob = df$prediction.prob
    all.pred = df$prediction.label

    confusion.mat = confusionMatrix(all.pred, reference = all.gt)

    bal.acc = confusion.mat$byClass["Balanced Accuracy"]
    AUC = pROC::auc(all.gt, all.prob)

    # levels(all.pred)[levels(all.pred)=="False"] = 0
    # levels(all.pred)[levels(all.pred)=="True"] = 1
    levels(all.gt)[levels(all.gt)=="True"] = 1
    levels(all.gt)[levels(all.gt)=="False"] = 0

    all.gt = as.numeric_version(all.gt)
    all.gt = as.numeric(all.gt)

    # all.pred = as.numeric_version(all.pred)
    # all.pred = as.numeric(all.pred)

    MCC = mcc(all.gt, all.prob, cutoff = 0.5) 
    
    if(is.nan(MCC))
    {
      MCC = 0
    }
    
    # MCC = -1*MCC # it seems that the sign is opposite from the result in sklearn

    all.auc = append(all.auc,AUC)
    all.mcc = append(all.mcc,MCC)
    all.bal.acc = append(all.bal.acc,bal.acc)
    all.test.rels = append(all.test.rels,f)

    # print(paste0('finished ',release))
  }
  
  result.df = data.frame(all.auc,all.mcc,all.bal.acc)
  
  all.test.rels = str_replace(all.test.rels, "-6-epochs.csv", "")
  all.test.rels = str_replace(all.test.rels, ".csv", "")
  
  result.df$release = all.test.rels
  result.df$technique = method.name
  
  return(result.df)
}

bi.lstm.prediction.dir = "../output/prediction/Bi-LSTM-6-epochs/"
cnn.prediction.dir = "../output/prediction/CNN/"

dbn.prediction.dir = "../output/prediction/DBN/"
lr.prediction.dir = "../output/prediction/LR/"
# deepline.dp.prediction.dir = "../output/prediction/DeepLineDP/rebalancing-adaptive-ratio2-lowercase-with-comment-50-dim-6-epochs/"

bi.lstm.result = get.file.level.eval.result(bi.lstm.prediction.dir, "Bi.LSTM") # some MCC are NaN
cnn.result = get.file.level.eval.result(cnn.prediction.dir, "CNN") # some MCC are NaN
dbn.result = get.file.level.eval.result(dbn.prediction.dir, "DBN")
lr.result = get.file.level.eval.result(lr.prediction.dir, "LR")
deepline.dp.result = get.file.level.eval.result(prediction_dir, "DeepLineDP")


# write.csv(deepline.dp.result, 'DeepLineDP-RQ2-result.csv')

all.result = rbind(bi.lstm.result, cnn.result, dbn.result, lr.result, deepline.dp.result)

names(all.result) = c("AUC","MCC","Balance.Accuracy","Release", "Technique")

auc.result = select(all.result, c("Technique","AUC"))
auc.result = preprocess(auc.result,FALSE)
auc.result[auc.result$variable=="Bi.LSTM", "variable"] = "Bi-LSTM"

mcc.result = select(all.result, c("Technique","MCC"))
mcc.result = preprocess(mcc.result,FALSE)
mcc.result[mcc.result$variable=="Bi.LSTM", "variable"] = "Bi-LSTM"

bal.acc.result = select(all.result, c("Technique","Balance.Accuracy"))
bal.acc.result = preprocess(bal.acc.result,FALSE)
bal.acc.result[bal.acc.result$variable=="Bi.LSTM", "variable"] = "Bi-LSTM"

# ggplot(auc.result, aes(x=reorder(variable, -value, FUN=median), y=value)) + geom_boxplot() + facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + ylab("AUC") + xlab("")
# ggsave(paste0(save.fig.dir,"file-AUC_new.pdf"),width=4,height=2.5)

# ggplot(bal.acc.result, aes(x=reorder(variable, value, FUN=median), y=value)) + geom_boxplot() + facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + ylab("Balance Accuracy") + xlab("")
# ggsave(paste0(save.fig.dir,"file-Balance_Accuracy_new.pdf"),width=4,height=2.5)

ggplot(mcc.result, aes(x=reorder(variable, value, FUN=median), y=value)) + geom_boxplot()  + facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + ylab("MCC") + xlab("")
# ggsave(paste0(save.fig.dir, "file-MCC_new.pdf"),width=4,height=2.5)


# auc.plot = ggplot(all.result, aes(x=reorder(Technique, -AUC, FUN=median), y=AUC)) + geom_boxplot() + ylab("AUC") + xlab("")
# 
# mcc.plot = ggplot(all.result, aes(x=reorder(Technique, -MCC, FUN=median), y=MCC)) + geom_boxplot() + ylab("MCC") + xlab("")
# 
# bal.acc.plot = ggplot(all.result, aes(x=reorder(Technique, -Balance.Accuracy), y=Balance.Accuracy)) + geom_boxplot() + ylab("Balance Accuracy") + xlab("")

# --------------------------------------------------------------------------#

# RQ3

## prepare data for baseline
line.ground.truth = select(df_all,  project, train, test, filename, file.level.ground.truth, prediction.prob, line.number, line.level.ground.truth)
line.ground.truth = filter(line.ground.truth, file.level.ground.truth == "True" & prediction.prob >= 0.5)
line.ground.truth = distinct(line.ground.truth)

get.line.metrics.result = function(baseline.df, cur.df.file)
{
  baseline.df.with.ground.truth = merge(baseline.df, cur.df.file, by=c("filename", "line.number"))
  
  # View(baseline.df.with.ground.truth)
  
  sorted = baseline.df.with.ground.truth %>% group_by(filename) %>% arrange(-line.score, .by_group = TRUE) %>% mutate(order = row_number())
  
  IFA = sorted %>% filter(line.level.ground.truth == "True") %>% group_by(filename)  %>% top_n(1, -order)
  
  ifa.list = IFA$order
  
  total_true = sorted %>% group_by(filename) %>% summarize(total_true = sum(line.level.ground.truth == "True"))
  
  #Recall20%LOC
  recall20LOC = sorted %>% group_by(filename) %>% mutate(effort = round(order/n(),digits = 2 )) %>% filter(effort <= 0.2) %>%
    summarize(correct_pred = sum(line.level.ground.truth == "True")) %>%
    merge(total_true) %>% mutate(recall20LOC = correct_pred/total_true)
  
  # summary(recall20LOC$recall20LOC)
  
  recall.list = recall20LOC$recall20LOC
  
  #Effort20%Recall
  effort20Recall = sorted %>% merge(total_true) %>% group_by(filename) %>% mutate(cummulative_correct_pred = cumsum(line.level.ground.truth == "True"), recall = round(cumsum(line.level.ground.truth == "True")/total_true, digits = 2)) %>%
    summarise(effort20Recall = sum(recall <= 0.2)/n())
  
  # summary(effort20Recall$effort20Recall)
  
  effort.list = effort20Recall$effort20Recall
  
  # print(length(ifa.list))
  # print(length(recall.list))
  # print(length(effort.list))
  
  result.df = data.frame(ifa.list, recall.list, effort.list)
  
  return(result.df)
}

all_eval_releases = c('activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 
                      'camel-2.10.0', 'camel-2.11.0' , 
                      'derby-10.5.1.1' , 'groovy-1_6_BETA_2' , 'hbase-0.95.2', 
                      'hive-0.12.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1',  
                      'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3')

error.prone.result.dir = '../output/ErrorProne_result/'
ngram.result.dir = '../output/n_gram_result/'

# note: remove pred-score column before uploading code to github (check this carefully na)
rf.result.dir = '../output/RF-line-level-result/'

n.gram.result.df = NULL
error.prone.result.df = NULL
rf.result.df = NULL # for predict()
# rf.result.prob.df = NULL # for predict_proba

## get result from baseline
for(rel in all_eval_releases)
{
  n.gram.result = read.csv(paste0(ngram.result.dir,rel,'-line-lvl-result.txt'), quote = "")
  
  error.prone.result = read.csv(paste0(error.prone.result.dir,rel,'-line-lvl-result.txt'),quote="")
  

  levels(error.prone.result$EP_prediction_result)[levels(error.prone.result$EP_prediction_result)=="False"] = 0
  levels(error.prone.result$EP_prediction_result)[levels(error.prone.result$EP_prediction_result)=="True"] = 1
  
  error.prone.result$EP_prediction_result = as.numeric(as.numeric_version(error.prone.result$EP_prediction_result))
  
  names(error.prone.result) = c("filename","test","line.number","line.score")
  
  n.gram.result = select(n.gram.result, "filename", "line.number",  "line.score")
  n.gram.result = distinct(n.gram.result)
  names(n.gram.result) = c("filename", "line.number", "line.score")
  
  
  
  cur.df.file = filter(line.ground.truth, test==rel)
  cur.df.file = select(cur.df.file, filename, line.number, line.level.ground.truth)
  
  
  n.gram.eval.result = get.line.metrics.result(n.gram.result, cur.df.file)
  
  print('finish n-gram')
  
  error.prone.eval.result = get.line.metrics.result(error.prone.result, cur.df.file)
  
  print('finish error prone')
  
  n.gram.result.df = rbind(n.gram.result.df, n.gram.eval.result)
  error.prone.result.df = rbind(error.prone.result.df, error.prone.eval.result)
  
  rf.result = read.csv(paste0(rf.result.dir,rel,'-line-lvl-result.csv'))
  rf.result = select(rf.result, "filename", "line_number","line.score.pred")
  names(rf.result) = c("filename", "line.number", "line.score")
  rf.eval.result = get.line.metrics.result(rf.result, cur.df.file)
  rf.result.df = rbind(rf.result.df, rf.eval.result)
  
  print('finish RF')
  
  print(paste0('finished ', rel))
  
  # rf.result.prob = select(rf.result, "filename", "line_number","line.score.prob")
  # names(rf.result.prob) = c("filename", "line.number", "line.score")
  # rf.eval.result = get.line.metrics.result(rf.result.prob, cur.df.file)
  # rf.result.prob.df = rbind(rf.result.prob.df, rf.eval.result)
  # missing Derby in RF-line-level here... just skip it
  # try({
  #   
  #   rf.result = read.csv(paste0(rf.result.dir,rel,'-line-lvl-result.csv'))
  #   rf.result = select(rf.result, "filename", "line_number","score")
  #   names(rf.result) = c("filename", "line.number", "line.score")
  #   rf.eval.result = get.line.metrics.result(rf.result, cur.df.file)
  #   rf.result.df = rbind(rf.result.df, rf.eval.result)
  #   
  # }, silent = TRUE)
}

#Force attention score of comment line is 0
df_all[df_all$is.comment.line == "True",]$token.attention.score = 0
# df_all[df_all$is.comment.line == "True",]$line.attention.score = 0
#Force attention score of java keywords is 0
# df_all[df_all$token %in% java_keywords$V1,]$token.attention.score = 0

#RQ3

# only needed for token attention
# top100 <- df_all_copy %>% filter( is.comment.line=="False"  & file.level.ground.truth=="True" & prediction.label=="True" ) %>%
#   group_by(test, filename) %>% top_n(100, token.attention.score) %>% select("project","train","test","filename","token") %>% distinct()
tmp.top.k = get.top.k.tokens(df_all, 1500)

merged_df_all = merge(df_all, tmp.top.k, by=c('project', 'train', 'test', 'filename', 'token'), all.x = TRUE)

merged_df_all[is.na(merged_df_all$flag),]$token.attention.score = 0



# df.top100.tokens = df_all_copy %>% group_by(project, train,test,filename,token) %>% filter(token %in% tmp.top100$token)
# df.other.tokens = df_all_copy %>% group_by(project, train,test,filename,token) %>% filter(! token %in% tmp.top100$token)

## use top-k tokens 
sum_line_attn = merged_df_all %>% filter(file.level.ground.truth == "True" & prediction.label == "True") %>% group_by(test, filename,is.comment.line, file.level.ground.truth, prediction.label, line.number, line.level.ground.truth) %>%
  summarize(attention_score = sum(token.attention.score), num_tokens = n())

# sum_line_attn = df_all %>% filter(file.level.ground.truth == "True" & prediction.label == "True") %>% group_by(test, filename,is.comment.line, file.level.ground.truth, prediction.label, line.number, line.level.ground.truth) %>%
#   summarize(attention_score = sum(token.attention.score), num_tokens = n())
# 


sorted = sum_line_attn %>% group_by(test, filename) %>% arrange(-attention_score, .by_group=TRUE) %>% mutate(order = row_number())

# only needed for line attention
# filtered_df_all = df_all %>% filter(file.level.ground.truth == "True" & prediction.label=="True")
# sorted = filtered_df_all %>% group_by(test,filename) %>% arrange(-line.attention.score, .by_group = TRUE) %>% mutate(order = row_number())

# calculate IFA
IFA = sorted %>% filter(line.level.ground.truth == "True") %>% group_by(test, filename) %>% top_n(1, -order)
# summary(IFA$order)
total_true = sorted %>% group_by(test, filename) %>% summarize(total_true = sum(line.level.ground.truth == "True"))

# calculateRecall20%LOC
recall20LOC = sorted %>% group_by(test, filename) %>% mutate(effort = round(order/n(),digits = 2 )) %>% filter(effort <= 0.2) %>%
  summarize(correct_pred = sum(line.level.ground.truth == "True")) %>%
  merge(total_true) %>% mutate(recall20LOC = correct_pred/total_true)
# summary(recall20LOC$recall20LOC)

# calculate Effort20%Recall
effort20Recall = sorted %>% merge(total_true) %>% group_by(test, filename) %>% mutate(cummulative_correct_pred = cumsum(line.level.ground.truth == "True"), recall = round(cumsum(line.level.ground.truth == "True")/total_true, digits = 2)) %>%
  summarise(effort20Recall = sum(recall <= 0.2)/n())
# summary(effort20Recall$effort20Recall)


# write.csv(IFA, 'DeepLineDP-RQ3-IFA.csv')
# write.csv(recall20LOC, 'DeepLineDP-RQ3-Recall.csv')
# write.csv(effort20Recall, 'DeepLineDP-RQ3-Effort.csv')

## prepare data for plotting
deeplinedp.ifa = IFA$order
deeplinedp.recall = recall20LOC$recall20LOC
deeplinedp.effort = effort20Recall$effort20Recall

deepline.dp.line.result = data.frame(deeplinedp.ifa, deeplinedp.recall, deeplinedp.effort)

names(rf.result.df) = c("IFA", "Recall20%LOC", "Effort@20%Recall")
# names(rf.result.prob.df) = c("IFA", "Recall20%LOC", "Effort@20%Recall")
names(n.gram.result.df) = c("IFA", "Recall20%LOC", "Effort@20%Recall")
names(error.prone.result.df)  = c("IFA", "Recall20%LOC", "Effort@20%Recall")
names(deepline.dp.line.result) = c("IFA", "Recall20%LOC", "Effort@20%Recall")

rf.result.df$technique = 'RF'
# rf.result.prob.df$technique = 'RF-Prob'
n.gram.result.df$technique = 'N.gram'
error.prone.result.df$technique = 'ErrorProne'
deepline.dp.line.result$technique = 'DeepLineDP'

all.line.result = rbind(rf.result.df, n.gram.result.df, error.prone.result.df, deepline.dp.line.result)

recall.result.df = select(all.line.result, c('technique', 'Recall20%LOC'))
ifa.result.df = select(all.line.result, c('technique', 'IFA'))
effort.result.df = select(all.line.result, c('technique', 'Effort@20%Recall'))

recall.result.df = preprocess(recall.result.df, FALSE)
ifa.result.df = preprocess(ifa.result.df, TRUE)
effort.result.df = preprocess(effort.result.df, TRUE)

ggplot(recall.result.df, aes(x=reorder(variable, -value, FUN=median), y=value)) + geom_boxplot() + facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + ylab("Recall@Top20%LOC") + xlab("")
ggsave(paste0(save.fig.dir,"file-Recall@Top20LOC_new.pdf"),width=4,height=2.5)

ggplot(effort.result.df, aes(x=reorder(variable, value, FUN=median), y=value)) + geom_boxplot() + facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + ylab("Effort@Top20%Recall") + xlab("")
ggsave(paste0(save.fig.dir,"file-Effort@Top20Recall_new.pdf"),width=4,height=2.5)

ggplot(ifa.result.df, aes(x=reorder(variable, value, FUN=median), y=value)) + geom_boxplot()  + coord_cartesian(ylim=c(0,175)) + facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + ylab("IFA") + xlab("")
ggsave(paste0(save.fig.dir, "file-IFA_new.pdf"),width=4,height=2.5)

# recall.plot = ggplot(all.line.result, aes(x=reorder(technique, -`Recall20%LOC`, FUN=median), y=`Recall20%LOC`)) + geom_boxplot()  + ylab("Recall@Top20LOC") + xlab("")
# # ggsave(paste0("file-Recall@Top20LOC_new.pdf"),width=4,height=2.5)
# 
# effort.plot = ggplot(all.line.result, aes(x=reorder(technique, `Effort@20%Recall`, FUN=median), y=`Effort@20%Recall`)) + geom_boxplot() +  ylab("Effort@Top20Recall") + xlab("")
# # ggsave(paste0("file-Effort@Top20Recall_new.pdf"),width=4,height=2.5)
# 
# ifa.plot = ggplot(all.line.result, aes(x=reorder(technique, IFA, FUN=median), y=IFA)) + geom_boxplot()  + coord_cartesian(ylim=c(0,200))  + ylab("IFA") + xlab("")
# ggsave(paste0("file-IFA_new.pdf"),width=4,height=2.5)


## for getting result of each project
deepline.dp.result$project = c("activemq", "activemq", "activemq", "camel", "camel", "derby", "groovy", "hbase", "hive", "jruby", "jruby", "lucene", "lucene", "wicket")

file.level.by.project = deepline.dp.result %>% group_by(project) %>% summarise(median.AUC = median(all.auc), median.MCC = median(all.mcc), median.bal.acc = median(all.bal.acc))

names(file.level.by.project) = c("project", "AUC", "MCC", "Balance Accurracy")

df_loc = df_all_correct_predict %>% select(test, filename, line.number) %>% distinct() %>% group_by(test, filename) %>% mutate(loc = n()) %>% select(test, filename, loc) %>% distinct()

IFA$project = str_replace(IFA$test, '-.*','')
recall20LOC$project = str_replace(recall20LOC$test, '-.*','')
recall20LOC$project = as.factor(recall20LOC$project)
effort20Recall$project = str_replace(effort20Recall$test, '-.*','')

ifa.each.project = IFA %>% group_by(project) %>% summarise(median.by.project = median(order))
recall.each.project = recall20LOC %>% group_by(project) %>% summarise(median.by.project = median(recall20LOC))
effort.each.project = effort20Recall %>% group_by(project) %>% summarise(median.by.project = median(effort20Recall))

line.level.all.median.by.project = data.frame(ifa.each.project$project, ifa.each.project$median.by.project, recall.each.project$median.by.project, effort.each.project$median.by.project)

names(line.level.all.median.by.project) = c("project", "IFA", "recall", "effort")

recall.with.loc = merge(recall20LOC, df_loc, by=c("test","filename"))

########################################################################

# for analyzing line-level result
# will remove later...


oov.count.dir = '../output/oov_count/'
oov.count.files = list.files(oov.count.dir)
all.oov.df = NULL

token.stat.dir = '../output/token_freq_stat/'
token.stat.files = list.files(token.stat.dir)
all.token.stat.df = NULL

for(f in token.stat.files)
{
  token.stat.df = read.csv(paste0(token.stat.dir,f))
  all.token.stat.df = rbind(all.token.stat.df, token.stat.df)
}

names(all.token.stat.df) = c("filename",    "code_line",   "line.number", "line.label",  "tok.freq.avg",    "tok.freq.median")

sorted.with.token.stat = merge(sorted, all.token.stat.df, by=c('filename','line.number'))

sorted.with.token.stat = select(sorted.with.token.stat, c("filename", "line.number", "test", "line.level.ground.truth", "attention_score", "num_tokens", "order", "code_line", "line.label", "tok.freq.avg",  "tok.freq.median"))

sorted.with.token.stat$project = str_replace(sorted.with.token.stat$test,'-.*', '')

sample.sorted.with.token.stat = filter(sorted.with.token.stat, project=="wicket", filename=="wicket-core/src/main/java/org/apache/wicket/request/mapper/CryptoMapper.java")


ggplot(data=sample.sorted.with.token.stat, aes(x=tok.freq.avg, y=attention_score, group=line.level.ground.truth)) + geom_point(aes(color=line.level.ground.truth, shape=line.level.ground.truth))



#---------------------------------------------------------------------#

for(f in oov.count.files)
{
  oov.df = read.csv(paste0(oov.count.dir,f))
  all.oov.df = rbind(all.oov.df, oov.df)
}





sorted.with.oov = merge(sorted, all.oov.df, by=c('filename','line.number'))

sorted.with.oov = select(sorted.with.oov, c("filename", "line.number", "test", "line.level.ground.truth", "attention_score", "num_tokens", "order", "code_line", "line.label", "oov_count"))

sorted.with.oov$project = str_replace(sorted.with.oov$test,'-.*', '')

sorted.with.oov$oov.ratio = sorted.with.oov$oov_count/sorted.with.oov$num_tokens

sorted.with.oov$avg.line.score = sorted.with.oov$attention_score/sorted.with.oov$num_tokens

sorted.with.oov$non.oov.count = sorted.with.oov$num_tokens - sorted.with.oov$oov_count

sorted.with.oov$non.oov.ratio = sorted.with.oov$non.oov.count / sorted.with.oov$num_tokens

sample.sorted.with.oov = filter(sorted.with.oov, project=="wicket", filename=="wicket-core/src/main/java/org/apache/wicket/markup/html/form/RadioChoice.java")

# clean.sample.sorted.with.oov = filter(sample.sorted.with.oov, line.label=="False")
# defect.sample.sorted.with.oov = filter(sample.sorted.with.oov, line.label=="True")

ggplot(data=sample.sorted.with.oov, aes(x=non.oov.ratio, y=attention_score, group=line.level.ground.truth)) + geom_point(aes(color=line.level.ground.truth, shape=line.level.ground.truth))

# ggplot(data=sample.sorted.with.oov, aes(x=oov_count, y=attention_score, group=line.level.ground.truth)) + geom_point(aes(color=line.level.ground.truth))

# ggplot(data=sample.sorted.with.oov, aes(x=num_tokens, y=attention_score, group=line.level.ground.truth)) + geom_point(aes(color=line.level.ground.truth, shape=line.level.ground.truth))

# ggplot(data=sample.sorted.with.oov, aes(x=oov.ratio, y=attention_score, group=line.level.ground.truth)) + geom_point(aes(color=line.level.ground.truth))


########################################################################
