//a small, pure combinatorial design.
//use for super simple FPGA placement,
//

//or for testing invertible logic.
//

module main (a, b, c, d, e, f);
	input wire a, b, c, d, e;
	output wire f;

	assign f = (a & b & c) ^ (d || e);

endmodule