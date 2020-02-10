/*
 * Written by: Sandeepa Kannangara, University of New South Wales, s.kannangara@unsw.edu.au
 * Part of code is from http://gibbslda.sourceforge.net/.
*/
package jsis_reverse;


public class Pair  implements Comparable<Pair> {
	public Object first;
	public Comparable second;
	public static boolean naturalOrder = false;
	
	public Pair(Object k, Comparable v){
		first = k;
		second = v;		
	}
	
	public Pair(Object k, Comparable v, boolean naturalOrder){
		first = k;
		second = v;
		Pair.naturalOrder = naturalOrder; 
	}
	
	public int compareTo(Pair p){
		if (naturalOrder)
			return this.second.compareTo(p.second);
		else return -this.second.compareTo(p.second);
	}
}
