package jsis_reverse;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;



public class LDA {
	
	public static void main(String[] args) {
		for(int i = 1; i <= 10; i++) {
			LDACmdOption option = new LDACmdOption();
			CmdLineParser parser = new CmdLineParser(option);
			args = new String[]{"-est", "-dfile", "data/posts", "-dir", "models/JSIS-Reverse/"+i+"/", "-itopics", "23", "-ytopics", "3", "-stopics", "2", "-twords", "20", "-alpha_i", "0.11", "-beta_i", "1", "-alpha_s", "1.25", "-beta_s", "1", "-alpha_y", "0.84", "-beta_y", "1"};
			//args = new String[] {"-inf", "-dir", "models/JEAS-Reverse/"+i, "-model", "model-final", "-niters", "1000", "-twords", "20", "-dfile", "test_new.dat"};
			
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
