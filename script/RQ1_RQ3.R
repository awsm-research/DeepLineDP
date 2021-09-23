
library(tidyverse)
library(gridExtra)

# library(MLmetrics)
library(ModelMetrics)

library(caret)

library(reshape2)
library(pROC)

# ---------------- Code for RQ1 and RQ3 (by AJ. Pick) -----------------------#

# current one
prediction_dir = '../output/prediction/DeepLineDP/rebalancing-adaptive-ratio2-lowercase-with-comment-50-dim-6-epochs/'
# prediction_dir = '../output/prediction/DeepLineDP/no-rebalancing-adaptive-ratio2-lowercase-with-comment-50-dim-14-epochs/'

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

    levels(all.pred)[levels(all.pred)=="False"] = 0
    levels(all.pred)[levels(all.pred)=="True"] = 1
    levels(all.gt)[levels(all.gt)=="True"] = 0
    levels(all.gt)[levels(all.gt)=="False"] = 1

    all.gt = as.numeric_version(all.gt)
    all.gt = as.numeric(all.gt)

    all.pred = as.numeric_version(all.pred)
    all.pred = as.numeric(all.pred)

    MCC = mcc(all.gt, all.pred, cutoff = 0.5) # it seems that the sign is opposite from the result in sklearn
    
    if(is.nan(MCC))
    {
      MCC = 0
    }
    
    MCC = -1*MCC

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

bi.lstm.prediction.dir = "../output/prediction/Bi-LSTM/"
cnn.prediction.dir = "../output/prediction/CNN/"

dbn.prediction.dir = "../output/prediction/DBN/"
lr.prediction.dir = "../output/prediction/LR/"
# deepline.dp.prediction.dir = "../output/prediction/DeepLineDP/rebalancing-adaptive-ratio2-lowercase-with-comment-50-dim-6-epochs/"

bi.lstm.result = get.file.level.eval.result(bi.lstm.prediction.dir, "Bi-LSTM") # some MCC are NaN
cnn.result = get.file.level.eval.result(cnn.prediction.dir, "CNN") # some MCC are NaN
dbn.result = get.file.level.eval.result(dbn.prediction.dir, "DBN")
lr.result = get.file.level.eval.result(lr.prediction.dir, "LR")
deepline.dp.result = get.file.level.eval.result(prediction_dir, "DeepLineDP")


# write.csv(deepline.dp.result, 'DeepLineDP-RQ2-result.csv')

all.result = rbind(bi.lstm.result, cnn.result, dbn.result, lr.result, deepline.dp.result)

names(all.result) = c("AUC","MCC","Balance.Accuracy","Release", "Technique")

auc.plot = ggplot(all.result, aes(x=reorder(Technique, -AUC, FUN=median), y=AUC)) + geom_boxplot() + ylab("AUC") + xlab("")

mcc.plot = ggplot(all.result, aes(x=reorder(Technique, -MCC, FUN=median), y=MCC)) + geom_boxplot() + ylab("MCC") + xlab("")

bal.acc.plot = ggplot(all.result, aes(x=reorder(Technique, -Balance.Accuracy), y=Balance.Accuracy)) + geom_boxplot() + ylab("Balance Accuracy") + xlab("")

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

n.gram.result.df = NULL
error.prone.result.df = NULL

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
  
  # n.gram.result.with.ground.truth = merge(n.gram.result, cur.df.file, by=c("filename", "line.number"))
  
  n.gram.eval.result = get.line.metrics.result(n.gram.result, cur.df.file)
  error.prone.eval.result = get.line.metrics.result(error.prone.result, cur.df.file)
  
  n.gram.result.df = rbind(n.gram.result.df, n.gram.eval.result)
  error.prone.result.df = rbind(error.prone.result.df, error.prone.eval.result)
  
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

names(n.gram.result.df) = c("IFA", "Recall20%LOC", "Effort@20%Recall")
names(error.prone.result.df)  = c("IFA", "Recall20%LOC", "Effort@20%Recall")
names(deepline.dp.line.result) = c("IFA", "Recall20%LOC", "Effort@20%Recall")

n.gram.result.df$technique = 'N-gram'
error.prone.result.df$technique = 'ErrorProne'
deepline.dp.line.result$technique = 'DeepLineDP'

all.line.result = rbind(n.gram.result.df, error.prone.result.df, deepline.dp.line.result)

recall.plot = ggplot(all.line.result, aes(x=reorder(technique, -`Recall20%LOC`, FUN=median), y=`Recall20%LOC`)) + geom_boxplot()  + ylab("Recall@Top20LOC") + xlab("")
# ggsave(paste0("file-Recall@Top20LOC_new.pdf"),width=4,height=2.5)

effort.plot = ggplot(all.line.result, aes(x=reorder(technique, `Effort@20%Recall`, FUN=median), y=`Effort@20%Recall`)) + geom_boxplot() +  ylab("Effort@Top20Recall") + xlab("")
# ggsave(paste0("file-Effort@Top20Recall_new.pdf"),width=4,height=2.5)

ifa.plot = ggplot(all.line.result, aes(x=reorder(technique, IFA, FUN=median), y=IFA)) + geom_boxplot()  + coord_cartesian(ylim=c(0,200))  + ylab("IFA") + xlab("")
# ggsave(paste0("file-IFA_new.pdf"),width=4,height=2.5)


## for getting result of each project
deepline.dp.result$project = c("activemq", "activemq", "activemq", "camel", "camel", "derby", "groovy", "hbase", "hive", "jruby", "jruby", "lucene", "lucene", "wicket")

file.level.by.project = deepline.dp.result %>% group_by(project) %>% summarise(median.AUC = median(all.auc), median.MCC = median(all.mcc), median.bal.acc = median(all.bal.acc))

names(file.level.by.project) = c("project", "AUC", "MCC", "Balance Accurracy")



IFA$project = str_replace(IFA$test, '-.*','')
recall20LOC$project = str_replace(recall20LOC$test, '-.*','')
effort20Recall$project = str_replace(effort20Recall$test, '-.*','')

ifa.each.project = IFA %>% group_by(project) %>% summarise(median.by.project = median(order))
recall.each.project = recall20LOC %>% group_by(project) %>% summarise(median.by.project = median(recall20LOC))
effort.each.project = effort20Recall %>% group_by(project) %>% summarise(median.by.project = median(effort20Recall))

line.level.all.median.by.project = data.frame(ifa.each.project$project, ifa.each.project$median.by.project, recall.each.project$median.by.project, effort.each.project$median.by.project)

names(line.level.all.median.by.project) = c("project", "IFA", "recall", "effort")

