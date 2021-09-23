library(tidyverse)
library(gridExtra)
library(reshape2)

# library(MLmetrics)
library(ModelMetrics)

library(caret)
library(pROC)



projs = c('activemq', 'camel', 'derby', 'groovy', 'hbase', 'hive', 'jruby', 'lucene', 'wicket')

prediction.dir = '../output/prediction/DeepLineDP/cross-release-rebalancing-adaptive-ratio2-lowercase-with-comment-50-dim/'

get.file.level.metrics = function(df.file)
{
  all.gt = df.file$file.level.ground.truth
  all.prob = df.file$prediction.prob
  all.pred = df.file$prediction.label
  
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
  MCC = -1*MCC
  
  if(is.nan(MCC))
  {
    MCC = 0
  }
  
  eval.result = c(AUC, MCC, bal.acc)
  
  return(eval.result)
}

get.line.level.metrics = function(df_all)
{
  #Force attention score of comment line is 0
  df_all[df_all$is.comment.line == "True",]$token.attention.score = 0
  #Force attention score of java keywords is 0
  # df_all[df_all$token %in% java_keywords$V1,]$token.attention.score = 0
  
  #RQ3
  
  # only needed for token attention
  sum_line_attn = df_all %>% filter(file.level.ground.truth == "True" & prediction.label == "True") %>% group_by(filename,is.comment.line, file.level.ground.truth, prediction.label, line.number, line.level.ground.truth) %>%
    summarize(attention_score = sum(token.attention.score), num_tokens = n())
  sorted = sum_line_attn %>% group_by(filename) %>% arrange(-attention_score, .by_group=TRUE) %>% mutate(order = row_number())
  
  # only needed for line attention
  # filtered_df_all = df_all %>% filter(file.level.ground.truth == "True" & prediction.label=="True")
  # sorted = filtered_df_all %>% group_by(test,filename) %>% arrange(-line.attention.score, .by_group = TRUE) %>% mutate(order = row_number())
  
  # calculate IFA
  IFA = sorted %>% filter(line.level.ground.truth == "True") %>% group_by(filename) %>% top_n(1, -order)
  # summary(IFA$order)
  total_true = sorted %>% group_by(filename) %>% summarize(total_true = sum(line.level.ground.truth == "True"))
  
  # calculateRecall20%LOC
  recall20LOC = sorted %>% group_by(filename) %>% mutate(effort = round(order/n(),digits = 2 )) %>% filter(effort <= 0.2) %>%
    summarize(correct_pred = sum(line.level.ground.truth == "True")) %>%
    merge(total_true) %>% mutate(recall20LOC = correct_pred/total_true)
  # summary(recall20LOC$recall20LOC)
  
  # calculate Effort20%Recall
  effort20Recall = sorted %>% merge(total_true) %>% group_by(filename) %>% mutate(cummulative_correct_pred = cumsum(line.level.ground.truth == "True"), recall = round(cumsum(line.level.ground.truth == "True")/total_true, digits = 2)) %>%
    summarise(effort20Recall = sum(recall <= 0.2)/n())
  # summary(effort20Recall$effort20Recall)
  
  all.ifa = IFA$order
  all.recall = recall20LOC$recall20LOC
  all.effort = effort20Recall$effort20Recall
  
  result.df = data.frame(all.ifa, all.recall, all.effort)
  
  return(result.df)
}


med.auc = c()
med.mcc = c()
med.bal.acc = c()

med.ifa = c()
med.recall = c()
med.effort = c()


for(p in projs)
{
  actual.pred.dir = paste0(prediction.dir,p,'/')
  
  all.files = list.files(actual.pred.dir)
  
  all.auc = c()
  all.mcc = c()
  all.bal.acc = c()
  
  all.line.result = NULL
  
  
  for(f in all.files)
  {
    df = read.csv(paste0(actual.pred.dir,f))
    df = as_tibble(df)
    
    df.file = select(df, c(train, test, filename, file.level.ground.truth, prediction.prob, prediction.label))
    
    df.file = distinct(df.file)
    
    # print(summary(df.file))
    
    file.level.result = get.file.level.metrics(df.file)

    AUC = file.level.result[1]
    MCC = file.level.result[2]
    bal.acc = file.level.result[3]

    all.auc = append(all.auc, AUC)
    all.mcc = append(all.mcc, MCC)
    all.bal.acc = append(all.bal.acc, bal.acc)

    line.level.result = get.line.level.metrics(df)

    all.line.result = rbind(all.line.result, line.level.result)
    
    # break
    
    print(paste0('finished ',f))
    
  }
  
  # df.all.file.result = data.frame(all.files, all.auc, all.mcc, all.bal.acc)
  # 
  # df.all.file.result$project = p
  
  
  
  # names(all.result) = c("filename", "AUC", "MCC", "Balance Accuracy","IFA","Recall", "Effort", "train project")
  # 
  # df.all.result = rbind(df.all.result, all.result)
  
  med.auc = append(med.auc, median(all.auc))
  med.mcc = append(med.mcc, median(all.mcc))
  med.bal.acc = append(med.bal.acc, median(all.bal.acc))
  
  med.ifa = append(med.ifa, median(all.line.result$all.ifa))
  med.recall = append(med.recall, median(all.line.result$all.recall))
  med.effort = append(med.effort, median(all.line.result$all.effort))
  
  
  print(paste0('finished ',p))
  # break
}

cross.project.result.df = data.frame(projs, med.auc, med.bal.acc, med.mcc, med.recall, med.effort, med.ifa)

names(cross.project.result.df) = c("project", "AUC", "Balance Accuracy", "MCC", "Recall", "Effort", "IFA")
