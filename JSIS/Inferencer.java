package jsis;

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
			for (int i = 0; i < trnModel.I; i++){
				trnModel.theta_i[m][i] = (trnModel.ndi[m][i] + trnModel.alpha_i) / (trnModel.nd[m] + trnModel.I * trnModel.alpha_i);
			}
		}
				
		for (int m = 0; m < trnModel.M; m++){
			for (int s = 0; s < trnModel.S; s++){
				trnModel.theta_s[m][s] = (trnModel.nds[m][s] + trnModel.alpha_s) / (trnModel.nd[m] + trnModel.S * trnModel.alpha_s);
			}
		}
					
		for (int m = 0; m < trnModel.M; m++){
			for (int i = 0; i < trnModel.I; i++){
				for (int s = 0; s < trnModel.S; s++) {
					for (int y = 0; y < trnModel.Y; y++) {
						trnModel.theta_y[m][i][s][y] = (trnModel.ndisy[m][i][s][y] + trnModel.alpha_isy[i][s][y]) / (trnModel.ndis[m][i][s] + trnModel.alphaSum_is[i][s]);
					}
					
				}
				
			}
		}
	}
	
	protected void computeTrnPhi(){
		for (int i = 0; i < trnModel.I; i++){
			for (int w = 0; w < trnModel.V; w++){
				trnModel.phi_i[i][w] = (trnModel.nwi[w][i] + trnModel.beta_iw[i][w]) / (trnModel.ni[i] + trnModel.beta_isum[i]);
			}
		}
		
		for (int s = 0; s < trnModel.S; s++){
			for (int w = 0; w < trnModel.V; w++){
				trnModel.phi_s[s][w] = (trnModel.nws[w][s] + trnModel.beta_sw[s][w]) / (trnModel.ns[s] + trnModel.beta_ssum[s]);
			}
		}
		
		for (int y = 0; y < trnModel.Y; y++){
			for (int w = 0; w < trnModel.V; w++){
				trnModel.phi_y[y][w] = (trnModel.nwy[w][y] + trnModel.beta_yw[y][w]) / (trnModel.ny[y] + trnModel.beta_ysum[y]);
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
			newModel.saveModel("Inference" + "." + newModel.modelName);		
			
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
			
			


			newModel.nwi[_w][issue] -= 1;
			newModel.nws[_w][sentiment] -= 1;
			newModel.nwy[_w][stance] -= 1;
			newModel.ni[issue] -= 1;
			newModel.ns[sentiment] -= 1;
			newModel.ny[stance] -= 1;
			newModel.ndi[m][issue] -= 1;
			newModel.nds[m][sentiment] -= 1;
			newModel.ndis[m][issue][sentiment] -= 1;
			newModel.ndisy[m][issue][sentiment][stance] -= 1;
			newModel.nd[m] -= 1;
			
			
			double Ialpha = newModel.alpha_i * newModel.I;
			double Salpha = newModel.alpha_s * newModel.S;
					
			
			if(newModel.data.lid2gid.containsKey(_w)) {
				int w = newModel.data.lid2gid.get(_w);
				//do multinominal sampling via cumulative method
				for (int i = 0; i < newModel.I; i++){
					for (int s = 0; s < newModel.S; s++) {
						for (int y = 0; y < newModel.Y; y++) {
							newModel.p[i][s][y] = (trnModel.nwi[w][i] + newModel.nwi[_w][i] + newModel.beta_iw[i][_w])/(trnModel.ni[i] + newModel.ni[i] + newModel.beta_isum[i]) *
									(trnModel.nws[w][s] + newModel.nws[_w][s] + newModel.beta_sw[s][_w])/(trnModel.ns[s] + newModel.ns[s] + newModel.beta_ssum[s]) *
									(trnModel.nwy[w][y] + newModel.nwy[_w][y] + newModel.beta_yw[y][_w])/(trnModel.ny[y] + newModel.ny[y] + newModel.beta_ysum[y]) *
									(newModel.ndi[m][i] + newModel.alpha_i)/(newModel.nd[m] + Ialpha) *
									(newModel.nds[m][s] + newModel.alpha_s)/(newModel.nd[m] + Salpha) *
									(newModel.ndisy[m][i][s][y] + newModel.alpha_isy[i][s][y])/(newModel.ndis[m][i][s] + newModel.alphaSum_is[i][s]);
						}						
					}
					
					
				}
			} else {
				//do multinominal sampling via cumulative method
				for (int i = 0; i < newModel.I; i++){
					for (int s = 0; s < newModel.S; s++) {
						for (int y = 0; y < newModel.Y; y++) {
							newModel.p[i][s][y] = (newModel.nwi[_w][i] + newModel.beta_iw[i][_w])/(newModel.ni[i] + newModel.beta_isum[i]) *
									(newModel.nws[_w][s] + newModel.beta_sw[s][_w])/(newModel.ns[s] + newModel.beta_ssum[s]) *
									(newModel.nwy[_w][y] + newModel.beta_yw[y][_w])/(newModel.ny[y] + newModel.beta_ysum[y]) *
									(newModel.ndi[m][i] + newModel.alpha_i)/(newModel.nd[m] + Ialpha) *
									(newModel.nds[m][s] + newModel.alpha_s)/(newModel.nd[m] + Salpha) *
									(newModel.ndisy[m][i][s][y] + newModel.alpha_isy[i][s][y])/(newModel.ndis[m][i][s] + newModel.alphaSum_is[i][s]);
						}
						
					}
					
					
				}
			}
			
			
			
			
			// cumulate multinomial parameters
			for (int i = 0; i < newModel.I; i++){
				for (int s = 0; s < newModel.S; s++) {
					for (int y = 0; y < newModel.Y; y++) {
						if(y == 0) {
							if( s == 0) {
								if(i == 0) {
										continue;
								}
								else {
									newModel.p[i][s][y] += newModel.p[i - 1][newModel.S - 1][newModel.Y - 1];
								}
								
								
							}
							else {
								newModel.p[i][s][y] += newModel.p[i][s - 1][newModel.Y - 1];
							}
						}
						else {
							newModel.p[i][s][y] += newModel.p[i][s][y - 1];
							
							
						}
						
					}
					
						
						
				}
				
			}
			
			// scaled sample because of unnormalized p[]
			double u = Math.random() * newModel.p[newModel.I - 1][newModel.S - 1][newModel.Y - 1];
			
			boolean loopbreak = false;
			for (issue = 0; issue < newModel.I ; issue++){
				for (sentiment = 0; sentiment < newModel.S; sentiment++) {
					for (stance = 0; stance < newModel.Y; stance++) {
						if (newModel.p[issue][sentiment][stance] > u) {//sample topic w.r.t distribution p
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
			/*	
			if( issue == newModel.I) issue--;
			if ( sentiment == newModel.S) sentiment--;
			*/
			// add newly estimated z_i to count variables
			
			newModel.nwi[_w][issue] += 1;
			newModel.nws[_w][sentiment] += 1;
			newModel.nwy[_w][stance] += 1;
			newModel.ni[issue] += 1;
			newModel.ns[sentiment] += 1;
			newModel.ny[stance] += 1;
			newModel.ndi[m][issue] += 1;
			newModel.nds[m][sentiment] += 1;
			newModel.ndis[m][issue][sentiment] += 1;
			newModel.ndisy[m][issue][sentiment][stance] += 1;
			newModel.nd[m] += 1;
			
			int [] results = new int [3]; 
			
			results[0] = issue;
			results[1] = sentiment;
			results[2] = stance;
			
	 		return results;
			

		}
		
		protected void computeNewTheta(){
			for (int m = 0; m < newModel.M; m++){
				for (int i = 0; i < newModel.I; i++){
					newModel.theta_i[m][i] = (newModel.ndi[m][i] + newModel.alpha_i) / (newModel.nd[m] + newModel.I * newModel.alpha_i);
				}
			}
			
			
			for (int m = 0; m < newModel.M; m++){
				for (int s = 0; s < newModel.S; s++){
					newModel.theta_s[m][s] = (newModel.nds[m][s] + newModel.alpha_s) / (newModel.nd[m] + newModel.S * newModel.alpha_s);
				}
			}
						
			for (int m = 0; m < newModel.M; m++){
				for (int i = 0; i < newModel.I; i++){
					for (int s = 0; s < newModel.S; s++) {
						for (int y = 0; y < newModel.Y; y++) {
							newModel.theta_y[m][i][s][y] = (newModel.ndisy[m][i][s][y] + newModel.alpha_isy[i][s][y]) / (newModel.ndis[m][i][s] + newModel.alphaSum_is[i][s]);
						}
						
					}
					
				}
			}
		}
		
		protected void computeNewPhi(){
		
			
			for (int i = 0; i < newModel.I; i++){
				for (int w = 0; w < newModel.V; w++){
					Integer id = newModel.data.lid2gid.get(w);
					if (id != null) {
						newModel.phi_i[i][w] = (trnModel.nwi[id][i] +newModel.nwi[w][i] + newModel.beta_iw[i][w]) / (trnModel.ni[i] + newModel.ni[i] + newModel.beta_isum[i]);
					}else {
						newModel.phi_i[i][w] = (newModel.nwi[w][i] + newModel.beta_iw[i][w]) / (newModel.ni[i] + newModel.beta_isum[i]);
						
					}
					
				}
			}
			
			for (int s = 0; s < newModel.S; s++){
				for (int w = 0; w < newModel.V; w++){
					Integer id = newModel.data.lid2gid.get(w);
					if (id != null) {
						newModel.phi_s[s][w] = (trnModel.nws[id][s] +newModel.nws[w][s] + newModel.beta_sw[s][w]) / (trnModel.ns[s] + newModel.ns[s] + newModel.beta_ssum[s]);
					}else {
						newModel.phi_s[s][w] = (newModel.nws[w][s] + newModel.beta_sw[s][w]) / (newModel.ns[s] + newModel.beta_ssum[s]);

					}
					
				}
			}
			
			for (int y = 0; y < newModel.Y; y++){
				for (int w = 0; w < newModel.V; w++){
					Integer id = newModel.data.lid2gid.get(w);
					if (id != null) {
						newModel.phi_y[y][w] = (trnModel.nwy[id][y] +newModel.nwy[w][y] + newModel.beta_yw[y][w]) / (trnModel.ny[y] + newModel.ny[y] + newModel.beta_ysum[y]);
					}else {
						newModel.phi_y[y][w] = (newModel.nwy[w][y] + newModel.beta_yw[y][w]) / (newModel.ny[y] + newModel.beta_ysum[y]);

					}
					
				}
			}
		}
		
}
