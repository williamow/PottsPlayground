//a small logic function with dffs with two different clock specifications,
//forcing the correct use of constraints in the simulated annealing model

module main (a, b, c, d, clk);
	input wire a, clk;
	output reg b, c, d;
	always @(posedge clk) b <= ~a & d;
	always @(negedge clk) c <= a & b;
	always @(negedge clk) d <= d ^ a;
endmodule