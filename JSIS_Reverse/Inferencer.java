package jsis_reverse;

import java.io.File;

public class Inferencer {
	// Train model
		public Model trnModel;
		public Dictionary globalDict;
		private LDACmdOption option;
		
		private Model newModel;
		//public int niters = 100;

	//-----------------------------------------------------
	// Init method
	//-----------------------------------------------------
	public boolean init(LDACmdOption option){
		this.option = option;
		trnModel = new Model();
			
		if (!trnModel.initEstimatedModel(option))
			return false;		
			
		globalDict = trnModel.data.localDict;
		computeTrnTheta();
		computeTrnPhi();
		
			
		return true;
	}
	
	protected void computeTrnTheta(){
		for (int m = 0; m < trnModel.M; m++){
			for (int s = 0; s < trnModel.Y; s++) {
				for (int e = 0; e < trnModel.I; e++){
					trnModel.theta_i[m][s][e] = (trnModel.ndyi[m][s][e] + trnModel.alpha_yi[s][e]) / (trnModel.ndy[m][s] + trnModel.alphaSum_yi[s]);
				}
			}
			
		}
		for (int m = 0; m < trnModel.M; m++){
			for (int s = 0; s < trnModel.Y; s++) {
				for (int t = 0; t < trnModel.S; t++){
					trnModel.theta_s[m][s][t] = (trnModel.ndys[m][s][t] + trnModel.alpha_ys[s][t]) / (trnModel.ndy[m][s] + trnModel.S * trnModel.alphaSum_ys[s]);
				}
			}
			
		}
		
		for (int m = 0; m < trnModel.M; m++){
				for (int s = 0; s < trnModel.Y; s++) {
						trnModel.theta_y[m][s] = (trnModel.ndy[m][s] + trnModel.alpha_y) / (trnModel.nd[m] + trnModel.alpha_y * trnModel.Y);
					}
		}
	}
	
	protected void computeTrnPhi(){
		for (int e = 0; e < trnModel.I; e++){
			for (int w = 0; w < trnModel.V; w++){
				trnModel.phi_i[e][w] = (trnModel.nwi[w][e] + trnModel.beta_iw[e][w]) / (trnModel.ni[e] + trnModel.beta_isum[e]);
			}
		}
		
		for (int t = 0; t < trnModel.S; t++){
			for (int w = 0; w < trnModel.V; w++){
				trnModel.phi_s[t][w] = (trnModel.nws[w][t] + trnModel.beta_sw[t][w]) / (trnModel.ns[t] + trnModel.beta_ssum[t]);
			}
		}
		
		for (int s = 0; s < trnModel.Y; s++){
			for (int w = 0; w < trnModel.V; w++){
				trnModel.phi_y[s][w] = (trnModel.nwy[w][s] + trnModel.beta_yw[s][w]) / (trnModel.ny[s] + trnModel.beta_ysum[s]);
			}
		}
	}
	
	//inference new model ~ getting dataset from file specified in option
		public Model inference(){	
			//System.out.println("inference");
			
			newModel = new Model();
			if (!newModel.initNewModel(option, trnModel)) return null;
			
			trnModel.data.localDict.writeWordMap(option.dir + File.separator + "new"+option.wordMapFileName);
			
			System.out.println("Sampling " + newModel.niters + " iteration for inference!");
			
			for (newModel.liter = 1; newModel.liter <= newModel.niters; newModel.liter++){
				System.out.println("Iteration " + newModel.liter + " ...");
				
				// for all newz_i
				for (int m = 0; m < newModel.M; ++m){
					for (int n = 0; n < newModel.data.docs[m].length; n++){
						
						
						int [] results = new int [3];
						results = infSampling(m, n);
						
						newModel.i[m].set(n, results[0]);
						newModel.s[m].set(n, results[1]);
						newModel.y[m].set(n, results[2]);
					}
				}//end foreach new doc
				
			}// end iterations
			
			System.out.println("Gibbs sampling for inference completed!");		
			System.out.println("Saving the inference outputs!");
			
			computeNewTheta();
			computeNewPhi();
			newModel.liter--;
			newModel.saveModel(newModel.dfile + "." + newModel.modelName);		
			
			return newModel;
		}
		
		/**
		 * do sampling for inference
		 * m: document number
		 * n: word number?
		 */
		protected int[] infSampling(int m, int n){
			// remove z_i from the count variables
			int issue = newModel.i[m].get(n);
			int sentiment = newModel.s[m].get(n);
			int stance = newModel.y[m].get(n);
			int _w = newModel.data.docs[m].words[n];
			int w = newModel.data.lid2gid.get(_w);
			


			newModel.nwi[_w][issue] -= 1;
			newModel.nws[_w][sentiment] -= 1;
			newModel.nwy[_w][stance] -= 1;
			newModel.ni[issue] -= 1;
			newModel.ns[sentiment] -= 1;
			newModel.ny[stance] -= 1;
			newModel.ndyi[m][stance][issue] -= 1;
			newModel.ndys[m][stance][sentiment] -= 1;
			newModel.ndy[m][stance] -= 1;
			newModel.nd[m] -= 1;
			
			
			
			double Yalpha = newModel.alpha_y * newModel.Y;
			
		
			
			
				//do multinominal sampling via cumulative method
				for (int e = 0; e < newModel.I; e++){
					for (int t = 0; t < newModel.S; t++) {
						for (int s = 0; s < newModel.Y; s++) {
							newModel.p[e][t][s] = (trnModel.nwi[w][e] + newModel.nwi[_w][e] + newModel.beta_iw[e][_w])/(trnModel.ni[e] + newModel.ni[e] + newModel.beta_isum[e]) *
									(trnModel.nws[w][t] + newModel.nws[_w][t] + newModel.beta_sw[t][_w])/(trnModel.ns[t]  + newModel.ns[t] + newModel.beta_ssum[t]) *
									(trnModel.nwy[w][s] + newModel.nwy[_w][s] + newModel.beta_yw[s][_w])/(trnModel.ny[s] + newModel.ny[s] + newModel.beta_ysum[s]) *
									(newModel.ndyi[m][s][e] + newModel.alpha_yi[s][e])/(newModel.ndy[m][s] + newModel.alphaSum_yi[s]) *
									(newModel.ndys[m][s][t] + newModel.alpha_ys[s][t])/(newModel.ndy[m][s] + newModel.alphaSum_ys[s]) *
									(newModel.ndy[m][s] + newModel.alpha_y)/(newModel.nd[m] + Yalpha);
						}
					}
					
				}
			
				
			
			
			
			
			
			// cumulate multinomial parameters
			for (int e = 0; e < newModel.I; e++){
				for (int t = 0; t < newModel.S; t++) {
					for (int s = 0; s < newModel.Y; s++) {
						if( s == 0) {
							if(t == 0) {
								if(e == 0) {
									continue;
								}
								else {
									newModel.p[e][t][s] += newModel.p[e - 1][newModel.S - 1][newModel.Y - 1];
								}
							}
							else {
								newModel.p[e][t][s] += newModel.p[e][t - 1][newModel.Y - 1];
							}
							
							
						}
						else {
							
							newModel.p[e][t][s] += newModel.p[e][t][s - 1];
						}
						
					}
				}
			}
			
			// scaled sample because of unnormalized p[]
			double u = Math.random() * newModel.p[newModel.I - 1][newModel.S - 1][newModel.Y - 1];
			
			boolean loopbreak = false;
			for (issue = 0; issue < newModel.I ; issue++){
				for (sentiment = 0; sentiment < newModel.S ; sentiment++) {
					for (stance = 0; stance < newModel.Y; stance++) {
						if (newModel.p[issue][sentiment][stance] > u) {//sample sentiment w.r.t distribution p
							loopbreak = true;
							break;
						}
							
					}
					
					if(loopbreak) {
						break;
					}
				}
				if(loopbreak) {
					break;
				}
			}
				
			if( issue == newModel.I) issue--;
			if ( sentiment == newModel.S) sentiment--;
			if ( stance == newModel.Y) stance--;
			
			// add newly estimated z_i to count variables
			
			newModel.nwi[_w][issue] += 1;
			newModel.nws[_w][sentiment] += 1;
			newModel.nwy[_w][stance] += 1;
			newModel.ni[issue] += 1;
			newModel.ns[sentiment] += 1;
			newModel.ny[stance] += 1;
			newModel.ndyi[m][stance][issue] += 1;
			newModel.ndys[m][stance][sentiment] += 1;
			newModel.ndy[m][stance] += 1;
			newModel.nd[m] += 1;
			
			int [] results = new int [3]; 
			
			results[0] = issue;
			results[1] = sentiment;
			results[2] = stance;
			
	 		return results;
			

		}
		
		protected void computeNewTheta(){
			for (int e = 0; e < newModel.I; e++){
				for (int w = 0; w < newModel.V; w++){
					newModel.phi_i[e][w] = (newModel.nwi[w][e] + newModel.beta_iw[e][w]) / (newModel.ni[e] + newModel.beta_isum[e]);
				}
			}
			
			for (int t = 0; t < newModel.S; t++){
				for (int w = 0; w < newModel.V; w++){
					newModel.phi_s[t][w] = (newModel.nws[w][t] + newModel.beta_sw[t][w]) / (newModel.ns[t] + newModel.beta_ssum[t]);
				}
			}
			
			for (int s = 0; s < newModel.Y; s++){
				for (int w = 0; w < newModel.V; w++){
					newModel.phi_y[s][w] = (newModel.nwy[w][s] + newModel.beta_yw[s][w]) / (newModel.ny[s] + newModel.beta_ysum[s]);
				}
			}
		}
		
		protected void computeNewPhi(){
		
			
			for (int e = 0; e < newModel.I; e++){
				for (int w = 0; w < newModel.V; w++){
					
						newModel.phi_i[e][w] = (trnModel.nwi[w][e] + newModel.nwi[w][e] + newModel.beta_iw[e][w]) / (trnModel.ni[e] + newModel.ni[e] + newModel.beta_isum[e]);
					
					
				}
			}
			
			for (int t = 0; t < newModel.S; t++){
				for (int w = 0; w < newModel.V; w++){
					
						newModel.phi_s[t][w] = (trnModel.nws[w][t] +newModel.nws[w][t] + newModel.beta_sw[t][w]) / (trnModel.ns[t] + newModel.ns[t] + newModel.beta_ssum[t]);
					
					
				}
			}
			
			for (int s = 0; s < newModel.Y; s++){
				for (int w = 0; w < newModel.V; w++){
					
						newModel.phi_y[s][w] = (trnModel.nwy[w][s] +newModel.nwy[w][s] + newModel.beta_yw[s][w]) / (trnModel.ny[s] + newModel.ny[s] + newModel.beta_ysum[s]);
					
					
				}
			}
		}
		
}
