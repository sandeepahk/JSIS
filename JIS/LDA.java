package jis;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;



public class LDA {

	public static void main(String[] args) {
		for ( int i = 7; i <= 10; i++) {
			LDACmdOption option = new LDACmdOption();
			CmdLineParser parser = new CmdLineParser(option);
			args = new String[]{"-est", "-dfile", "data/posts", "-dir", "models/JIS/"+i+"/", "-itopics", "23", "-stopics", "2", "-twords", "20", "-alpha_i", "0.11", "-beta_i", "1", "-alpha_s", "1.25", "-beta_s", "1"};
			
			try {
				if (args.length == 0){
					showHelp(parser);
					return;
				}
				
				parser.parseArgument(args);
				
				if (option.est || option.estc){
					Estimator estimator = new Estimator();
					estimator.init(option);
					estimator.estimate();
				}
				else if (option.inf){
					Inferencer inferencer = new Inferencer();
					inferencer.init(option);
					
					Model newModel = inferencer.inference();
				
					
				}
			}
			catch (CmdLineException cle){
				System.out.println("Command line error: " + cle.getMessage());
				showHelp(parser);
				return;
			}
			catch (Exception e){
				System.out.println("Error in main: " + e.getMessage());
				e.printStackTrace();
				return;
			}

		}
		
	}
	
	public static void showHelp(CmdLineParser parser){
		System.out.println("LDA [options ...] [arguments...]");
		parser.printUsage(System.out);
	}
	

}
