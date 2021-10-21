import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.util.DoubleSummaryStatistics;
import java.util.HashMap;
import java.util.stream.Collectors;

import slp.core.counting.giga.GigaCounter;
import slp.core.lexing.Lexer;
import slp.core.lexing.code.JavaLexer;
import slp.core.lexing.runners.LexerRunner;
import slp.core.lexing.simple.WhitespaceLexer;
import slp.core.modeling.Model;
import slp.core.modeling.dynamic.CacheModel;
import slp.core.modeling.mix.MixModel;
import slp.core.modeling.ngram.JMModel;
import slp.core.modeling.runners.ModelRunner;
import slp.core.translating.Vocabulary;

public class n_gram_newest
{
	public static String root_dir = "./n_gram_data/";
	public static String result_dir = "./n_gram_result/";
	
	public static String all_dataset[] = {"activemq","camel","derby","groovy","hbase","hive",
											"jruby","lucene","wicket"};
	public static String all_train_releases[] = {"activemq-5.0.0","camel-1.4.0","derby-10.2.1.6","groovy-1_5_7","hbase-0.94.0",
	                 "hive-0.9.0","jruby-1.1","lucene-2.3.0","wicket-1.3.0-incubating-beta-1"};
	public static String all_eval_releases[][] = {{"activemq-5.1.0","activemq-5.2.0","activemq-5.3.0","activemq-5.8.0"},
	                 {"camel-2.9.0","camel-2.10.0","camel-2.11.0"}, 
	                 {"derby-10.3.1.4","derby-10.5.1.1"},
	                 {"groovy-1_6_BETA_1","groovy-1_6_BETA_2"}, 
	                 {"hbase-0.95.0","hbase-0.95.2"},
	                 {"hive-0.10.0","hive-0.12.0"}, 
	                 {"jruby-1.4.0","jruby-1.5.0","jruby-1.7.0.preview1"},
	                 {"lucene-2.9.0","lucene-3.0.0","lucene-3.1"}, 
	                 {"wicket-1.3.0-beta2","wicket-1.5.3"}};
	
	public static String all_releases[][] = {{"activemq-5.0.0","activemq-5.1.0","activemq-5.2.0","activemq-5.3.0","activemq-5.8.0"},
            {"camel-1.4.0","camel-2.9.0","camel-2.10.0","camel-2.11.0"}, 
            {"derby-10.2.1.6","derby-10.3.1.4","derby-10.5.1.1"},
            {"groovy-1_5_7","groovy-1_6_BETA_1","groovy-1_6_BETA_2"}, 
            {"hbase-0.94.0","hbase-0.95.0","hbase-0.95.2"},
            {"hive-0.9.0", "hive-0.10.0","hive-0.12.0"}, 
            {"jruby-1.1", "jruby-1.4.0","jruby-1.5.0","jruby-1.7.0.preview1"},
            {"lucene-2.3.0","lucene-2.9.0","lucene-3.0.0","lucene-3.1"}, 
            {"wicket-1.3.0-incubating-beta-1", "wicket-1.3.0-beta2","wicket-1.5.3"}};
	
	public static ModelRunner train_model(String train_release)
	{
		Map to_return = new HashMap();
//			File train = new File("./for n-gram model\\activemq-5.0.0");
		File train = new File(root_dir+train_release+"/src");
		Lexer lexer = new WhitespaceLexer();   // Use a Java lexer; if your code is already lexed, use whitespace or tokenized lexer
		LexerRunner lexerRunner = new LexerRunner(lexer, false);  // Don't model lines in isolation for code files.
																  // We will still get per-line, per-token entropies
		lexerRunner.setSentenceMarkers(true);  // Add start and end markers to the files
//		lexerRunner.setExtension("java");  // We only lex Java files

		Vocabulary vocabulary = new Vocabulary();  // Create an empty vocabulary
		
		Model model = new JMModel(6, new GigaCounter());  // Standard smoothing for code, giga-counter for large corpora
		model = MixModel.standard(model, new CacheModel());  // Use a simple cache model; see JavaRunner for more options
		ModelRunner modelRunner = new ModelRunner(model, lexerRunner, vocabulary); // Use above lexer and vocabulary
		modelRunner.learnDirectory(train);  // Teach the model all the data in "train"
		
//		System.out.println("done");
//		System.out.println(train.getPath());
		return modelRunner;
	}
	
	public static void predict_defective_lines(String train_release, String test_release, ModelRunner modelRunner) throws Exception
	{
		LexerRunner lexerRunner = modelRunner.getLexerRunner();
		
		StringBuilder sb = new StringBuilder();
		
//		sb.append("File-Name\tLine-Number\tN-Gram-Score\n");
		sb.append("train-release\ttest-release\tfile-name\tline-number\ttoken\ttoken-score\tline-score\n");
		
//		LexerRunner lexerRunner = (LexerRunner) object_map.get("lexer");
//		ModelRunner modelRunner = (ModelRunner) object_map.get("model");

		File test_java_dir = new File(root_dir + test_release+"/src/");
		File java_files[] = test_java_dir.listFiles();
		
		String line_num_path = root_dir + test_release+"/line_num/";
		
//		System.out.println(java_files[55].getName());
		
		// loop each file here...
		
		for(int j = 0; j<java_files.length; j++)
		{
			File test = java_files[j];
			
			String filename = test.getName();
			String filename_original = filename.replace("_", "/").replace(".txt", ".java");
			String linenum_filename = filename.replace(".txt", "_line_num.txt");
			
			List<String> linenum = FileUtils.readLines(new File(line_num_path+linenum_filename),"UTF-8");
			
//			System.out.println(linenum);
//			System.out.println(filename);
//			System.out.println(linenum_filename);

			List<List<Double>> fileEntropies = modelRunner.modelFile(test);
			List<List<String>> fileTokens = lexerRunner.lexFile(test)  // Let's also retrieve the tokens on each line
					.map(l -> l.collect(Collectors.toList()))
					.collect(Collectors.toList());
			
//			System.out.println(linenum.size() + " " + fileEntropies.size());
			
			for (int i = 0; i < linenum.size(); i++) {
				List<String> lineTokens = fileTokens.get(i);
				List<Double> lineEntropies = fileEntropies.get(i);
				
				String cur_line_num = linenum.get(i);
				
//				System.out.println(lineTokens);
//				System.out.println(lineEntropies);
				
				// First use Java's stream API to summarize entropies on this line
				// (see modelRunner.getStats for summarizing file or directory results)
				DoubleSummaryStatistics lineStatistics = lineEntropies.stream()
						.mapToDouble(Double::doubleValue)
						.summaryStatistics();
				double averageEntropy = lineStatistics.getAverage();
				
//				sb.append("test-release\tline-number\ttoken\ttoken-score\tline-score\n");
				
				for(int k = 0; k< lineTokens.size(); k++)
				{
					String tok = lineTokens.get(k);
					double tok_score = lineEntropies.get(k);
					
					if(tok == "<s>")
						continue;
					
					sb.append(train_release+"\t"+test_release+"\t"+filename_original+"\t"+cur_line_num+"\t"+tok+"\t"+tok_score+"\t"+averageEntropy+"\n");
				}
				
//				sb.append(filename+"\t"+(i+1)+"\t"+averageEntropy+"\n");
	//			System.out.println(java_code.get(i) + " " + averageEntropy);
				
				
//				break;
			}
//			System.out.println(sb.toString());
//			break;
		}
		FileUtils.write(new File(result_dir+test_release+"-line-lvl-result.txt"), sb.toString(),"UTF-8");			
		
	}
	
	public static void train_eval_model(int dataset_idx) throws Exception
	{
		String dataset_name = all_dataset[dataset_idx];
		String train_release = all_train_releases[dataset_idx];
		String eval_release[] = all_eval_releases[dataset_idx];
		
		String cur_all_rels[] = all_releases[dataset_idx];
		
		ModelRunner modelRunner = train_model(train_release);
		
		System.out.println("finish training model for " + dataset_name);
		
		for(int idx = 0; idx<eval_release.length-1; idx++)
		{
//			String train_release = cur_all_rels[rel];
//			String eval_release = cur_all_rels[rel+1];
			String rel = eval_release[idx];
//			ModelRunner modelRunner = train_model(train_release);
			predict_defective_lines(train_release, rel, modelRunner);

			System.out.println("finish "+rel+"\n");
			
//			break;

		}
		
	}
	
	public static void main( String[] args ) throws Exception
	{
		for(int a = 0; a<9; a++)
		{
			train_eval_model(a);
//			break;
		}
//		String rel = "activemq-5.2.0";
//		String train_release = "activemq-5.0.0";
//		ModelRunner modelRunner = train_model(train_release);
//		predict_defective_lines(rel, modelRunner);
	}
}
