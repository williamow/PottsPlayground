module tiny (a, b, c, d, clk);
	input wire a, clk;
	output reg b, c, d;
	always @(posedge clk) b <= ~a & d;
	always @(negedge clk) c <= a & b;
	always @(negedge clk) d <= d ^ a;
endmodule


module main (
	//interface to FTDI during teathered setup - slave SPI
	FTDI_cfg_cs, FTDI_mosi, FTDI_miso, FTDI_sck, 

	//interface to UWB radio, for transmitting data - master SPI
	UWB_CS, UWB_MOSI, UWB_MISO, UWB_SCK, 

	//four phase control of micromirror
	// Xp, Xn, Yp, Yn,

	//interface to ADC: FPGA provides a clock, conversions come in on an 8-bit data bus
	// ADC_data, 
	ADC_clk,

	//pwm driver for setting APD bias:
	// APD_pwm,

	clk16, clk2
	);

	//definitions for module connections
	input wire FTDI_cfg_cs, FTDI_mosi, FTDI_sck, UWB_MISO;
	output wire UWB_CS, UWB_MOSI, UWB_SCK, FTDI_miso;
	// output wire Xp, Xn, Yp, Yn;
	wire [7:0] ADC_data;
	assign ADC_data = 48;
	output wire ADC_clk;
	// output wire APD_pwm;
	input wire clk16, clk2;

	wire FTDI_UWB_cs;
	assign FTDI_UWB_cs = 1; //for now, set to 1, i.e. disable

	//clocks: internal oscillator and PLL are used to generate clocks at several frequencies (denoted in MHz)
	// wire clk2, clk16; //2 and 16 MHz
	// clocks the_clocks(clk2, clk16);

	assign ADC_clk = clk2;


	//=============================================================================================================
	//SPI slave:  4-byte transfers. Byte 1 is command, byte 2 is address, byte 3-4 is data.
	wire [7:0] sl_cmd, sl_addr; //sl, to indicate association with slave SPI
	wire [15:0] sl_data;
	reg [31:0] sl_response;
	wire cfg_miso; //output back to FTDI, not always connected - must be mux'd 
	SPI_serializer #(4, 5) cfg_slave(FTDI_cfg_cs, cfg_miso, FTDI_mosi, FTDI_sck, {sl_cmd, sl_addr, sl_data}, sl_response);

	//regular slave SPI: ==========================================================================================	
	//bank of config registers:
	reg [15:0] cfg_regs [3:0];

	//map config regs to their purposes:
	assign periodX = cfg_regs[0];
	assign periodY = cfg_regs[1];
	assign APD_pwm_period = cfg_regs[2][7:0];
	assign APD_pwm_duty = cfg_regs[3][7:0];
	// assign SPImaster_enable = cfg_regs[4][0]; //turns UWB data writing on and off, so that it can be bypassed for initial config

	always @ (posedge FTDI_cfg_cs) begin //actions for config SPI
		case (sl_cmd)
			0:  sl_response <= {sl_cmd, sl_addr, sl_data}; //readback
			1:	cfg_regs[sl_addr] <= sl_data; //write to register
			2:  sl_response <= cfg_regs[sl_addr]; //cfg reg readback
			3:  sl_response <= ADC_data; //for direct sampling from the ADC, for debug use
		endcase
	end

	// simple peripheral module instantiations ===============================================================
	
	// pwm gen to drive APD bias:
	wire [7:0] APD_pwm_period, APD_pwm_duty;
	// pwm_gen APD_bias(clk2, APD_pwm, APD_pwm_period, APD_pwm_duty);

	//mirror drive pattern generator:
	wire [15:0] timerX, timerY; //module outputs for phase stamping data
	wire [15:0] periodX, periodY; //frequency settings from configuration SPI
	// mirror_driver md(clk2, Xp, Xn, Yp, Yn, timerX, timerY, periodX, periodY);

	//FAKE:
	assign timerX = 1;
	assign timerY = 1;


	//data pathway ===============================================================
	// ADC -> FIFO -> packet writer -> SPI master -> UWB chip

	//data is first saved into a FIFO.
	//ADC is 'valid' only every other clock cycle (just to reduce data rate to 1/2 clock rate)
	reg FIFO_in_valid;
	always @(posedge clk2) FIFO_in_valid = ~FIFO_in_valid;
	wire FIFO_in_ready, FIFO_out_valid; //for our simple implementation, we are going to ignore these, i.e. assume always ready and always valid
	wire FIFO_out_ready;
	wire [7:0] FIFO_out_data;
	//fifo should only be big enough to hold half a packet's worth of data, i.e. 30 data points
	FIFO #(3, 8, 5) compressed_main(clk2, clk2, ADC_data, FIFO_in_valid, FIFO_in_ready, FIFO_out_data, FIFO_out_valid, FIFO_out_ready);

	//SPI master.  Accepts bytes 1 at a time to transfer, and bytes are recieved one at a time
	wire [7:0] master_tx_data, master_rx_data; //data to be transmitted, and data just recieved, respectively
	wire master_rx_valid; //indicates when the master_rx_data is actual data.  Actually, not using this now.
	wire master_CS, master_MOSI, master_SCK; //SPI interface. conditionally passes thru to UWB SPI
	SPI_master master(master_tx_data, master_rx_data, master_rx_valid, clk16, clk2, master_CS, master_MOSI, UWB_MISO, master_SCK);
	
	wire [7:0] prog_to_spi;
	wire prog_FIFO_ready, prog_FIFO_valid; // ready/valid signals to grab data from the data FIFO
	uwb_packet_writer prog(clk2, master_tx_data, FIFO_out_data, FIFO_out_ready, timerX, timerY);
	
	//SPI muxing: UWB to master, slave, and FTDI
	assign UWB_CS = FTDI_UWB_cs && master_CS; //either one pulls low
	assign UWB_SCK = (~FTDI_UWB_cs && FTDI_sck)  || (~master_CS && master_SCK);  //or of gated clocks
	assign UWB_MOSI = (~FTDI_UWB_cs && FTDI_mosi)  || (~master_CS && master_MOSI); //or of gated MOSI
	assign FTDI_miso = (~FTDI_UWB_cs && UWB_MISO)  || (~FTDI_cfg_cs && cfg_miso); //decides which slave drives MISO

endmodule

module uwb_packet_writer(
	input wire clk,

	//command or data to control the SPI master
	output reg [7:0] to_spi,

	//interface to FIFO that holds the ADC data
	input wire [7:0] data,
	output reg FIFO_ready,

	//mirror driver info for stamping packets
	input wire [15:0] timerX,
	input wire [15:0] timerY
	);

	//program counter.
	//Simple linear program, no branches.
	//period of 120 (since 60 data bytes per packet)
	//ends at PC=127 so that case matching below is easier to write
	reg [7:0] PC; 
	always @(posedge clk)
		if (PC==127) PC <= 7;
		else PC <= PC + 1;

	//what to send to SPI master:
	always @(posedge clk)
		casez (PC)
			20: to_spi <= 2; //start a two-byte transfer
			21: to_spi <= 31+64; //command for writing to control register
			22: to_spi <= 16; //set control register to trigger a transmission

			56: to_spi <= 2; //start a two-byte transfer
			57: to_spi <= 31+64; //command for writing to control register 0x1F
			58: to_spi <= 16; //set control register to trigger a transmission

			8'b00111110: to_spi <= 65; //a 65 byte transfer: control byte plus data
			8'b00111111: to_spi <= 63+64+128; //burst write command

			//sequence of packet data.
			//Read that case executes in priority order.
			//i.e. the timer values will have priority over the data even though they overlap
			8'b01000000: to_spi <= timerX[15:8];
			8'b01000001: to_spi <= timerX[7:0];
			8'b01000010: to_spi <= timerY[15:8];
			8'b01000011: to_spi <= timerY[7:0];			
			8'b01??????: to_spi <= data;

			default: to_spi <= 0;
		endcase


	//make sure that data is popped from the FIFO when it is used.
	//that means setting FIFO ready the clock cycle BEFORE the data is used.
	always @(posedge clk)
		casez (PC)

			8'b01000000: FIFO_ready <= 0;
			8'b01000001: FIFO_ready <= 0;
			8'b01000010: FIFO_ready <= 0;
			8'b01111111: FIFO_ready <= 0;			
			8'b01??????: FIFO_ready <= 1; //taking advantage of priority order here as well

			default: FIFO_ready <= 0;
		endcase
endmodule

module pwm_gen(
	input wire clk,
	output wire out,
	input wire [7:0] total_period,
	input wire [7:0] high_period
	);
	// by specifying both the total period and the high period,
	// many more levels are acheivable without sacraficing frequency

	reg [7:0] counter;

	assign out = (counter > high_period) ? 0 : 1;

	always @(posedge clk)
		if (counter == total_period) counter <= 0;
		else counter <= counter + 1;
endmodule

module mirror_driver(
	clk,
	Xp, Xn, Yp, Yn, //control signals to mirror
	timerX, timerY,  //output state of timers to be encoded into data stream
	periodX, periodY //period settings
	);

	input wire [15:0] periodY, periodX;

	input wire clk;
	output wire Xp, Xn, Yp, Yn;

	output reg [15:0] timerX, timerY;

	//conditionally wire output drives based on timer values
	assign Xp = (timerX > periodX/2) ? 1:0;
	assign Xn = (timerX > periodX/2) ? 0:1;
	assign Yp = (timerY > periodY/2) ? 1:0;
	assign Yn = (timerY > periodY/2) ? 0:1;

	initial begin
		timerX = 0;
		timerY = 0;
	end

	//two timers with slightly different periods, to create beat frequency (scanning frame rate)
	always @ (posedge clk) begin
		
		if (timerX == periodX) timerX <= 0; //period of timerX
		else timerX <= timerX + 1;

		if (timerY == periodY) timerY <= 0; //period of timerY
		else timerY <= timerY + 1;
	end
endmodule



module clocks(output reg clk2, output reg clk16);

	wire hfosc_clock;
	SB_HFOSC #(.CLKHF_DIV("0b00")) hfosc ( //0b00 for full 48 MHz operation, 0b01 for 24 MHz operation
	  .CLKHFPU(1'b1),
	  .CLKHFEN(1'b1),
	  .CLKHF(hfosc_clock)
	);

	//pll down to 32 MHz, which then drives the divider:
	wire clk32, locked;
	//f_out = f_in * (DIVF+1) / (2^DIVQ)
	SB_PLL40_CORE #(
		.FEEDBACK_PATH("SIMPLE"),
		.DIVR(4'b0010),		// DIVR =  2
		.DIVF(7'b0111111),	// DIVF = 63
		.DIVQ(3'b101),		// DIVQ =  5
		.FILTER_RANGE(3'b001)	// FILTER_RANGE = 1
	) uut (
		.LOCK(locked),
		.RESETB(1'b1),
		.BYPASS(1'b0),
		.REFERENCECLK(hfosc_clock),
		.PLLOUTCORE(clk32)
		);

	reg [7:0] divider;
	always @ (posedge clk32) begin 
		divider <= divider+1;
		clk2 <= divider[3];
		clk16 <= divider[0];
	end

endmodule


module FIFO #(parameter depth=2048, parameter width=8, parameter addr_bits=11) (
	input wire in_clk, 
	input wire out_clock, 
	input wire [width-1:0] IN_DATA, 
	input wire IN_VALID, 
	output wire IN_READY, 
	output wire [width-1:0] OUT_DATA, 
	output wire OUT_VALID, 
	input wire OUT_READY);
	//FIFO made from implicit BRAM.
	//Input and output are on different clocks, allowing the FIFO to function as an asynchronous interface.  However if async, does not use BRAM.

	reg [addr_bits-1:0] wr_ptr;
	reg [addr_bits-1:0] rd_ptr;

	reg [width-1:0] mem[depth-1:0];

	assign IN_READY = ~(rd_ptr == wr_ptr+1);
	assign OUT_VALID = ~(rd_ptr == wr_ptr);
	assign OUT_DATA = mem[rd_ptr];

	always @ (posedge in_clk) begin
		if (IN_VALID && IN_READY) begin
			mem[wr_ptr] <= IN_DATA;
			wr_ptr <= wr_ptr + 1;
		end
	end

	always @ (posedge out_clock) begin
		if (OUT_VALID && OUT_READY) begin
			rd_ptr <= rd_ptr + 1;
		end
	end
endmodule

module SPI_master(tx_data, rx_data, rx_valid, clk_fast, clk_slow, CS, MOSI, MISO, SCK);
	//SR10x0 samples our data on rising edge of SCK, which is negative edge of clk48;
	//we sample SR10x0 data on falling edge of SCK, i.e. positive edge of clk48.  Since SR10x0 recieves SCK after the FPGA does, we should still see the old value.
	reg [7:0] tx_count;
	reg [6:0] tx_databuf;
	reg [7:0] rx_databuf;
	wire [7:0] tx_interface;
	assign tx_interface = {tx_data[7], tx_databuf};  //MSB is first to be shifted out, so we assign it directly instead of through the buffer
	assign rx_data = {rx_databuf[6:0], MISO}; //LSB is last to be read; therefore it is sent directly to the output byte interface
	input wire clk_slow, clk_fast, MISO;
	input wire [7:0] tx_data; //provides phase information between clk48 and clk6
	output wire [7:0] rx_data;
	output reg CS;
	output wire SCK, rx_valid, MOSI;
	// assign SCK = (SCK_phase1 == SCK_phase2) && (~CS);
	assign SCK = (clk_fast) && (~CS);

	always @ (posedge clk_slow) begin
		if (tx_count == 0) begin
			tx_count <= tx_data;
			CS <= 1; //chip select not active this cycle
		end
		else begin
			tx_count <= tx_count - 1;
			CS <= 0; //chip select active
			tx_databuf <= tx_data[6:0];
		end
	end

	assign rx_valid = ~CS;

	reg [2:0] tx_index;
	assign MOSI = tx_interface[tx_index];

	always @ (posedge clk_fast) begin
		if (CS) tx_index <= 7;
		else tx_index <= tx_index - 1;
	end

	always @ (negedge clk_fast) rx_databuf <= {rx_databuf[6:0], MISO}; //make it a shift register, not an addressed thing
endmodule



module SPI_serializer #(parameter xfer_len=4, parameter addr_bits=5) (cs, ser_out, ser_in, sck, data_in, data_out);
	//Generic SPI element. During a transfer, DI data is filled and DO data is scanned out.

	//parameter xfer_len determines the size of the max transfer length,
	//and parameter addr_bits must be sufficient to index through all bits of the max transfer length

	//external logic must be added to make actual use of DI_DATA (an action on rising CS) and to set DO_DATA before a transfer.
	//Can be used as a part in either a slave or a master.
		//for master: another bit of logic in the same system orchestrates CS to initiate transfers.
		//for slave: CS is controlled by a master SPI from somewhere else.


	input wire [(8*xfer_len-1):0] data_out; //DO is an input, because the data to be serialized is created locally;
	output reg [(8*xfer_len-1):0] data_in;
	input wire ser_in, cs, sck;
    output wire ser_out;

    //scan chain for scan_in_bits
    //syntax note:  line ends only at semicolon; no begin / end are needed since there is only one action inside each block
    always @(posedge sck)
        if (~cs) data_in <= {data_in[(8*xfer_len-2):0], ser_in};

    //output register:  content of output reg set at the end of the preceding transfer.
    reg [addr_bits:0] out_bit_counter;
    assign ser_out = data_out[out_bit_counter];

    //pointer state for serialization
    always @(posedge sck, posedge cs) begin
        if (cs) out_bit_counter <= 8*xfer_len-1; //reset to highest bit
        else out_bit_counter <= out_bit_counter-1; //MSB is shifted out first; so bit counter decrements
    end
    
endmodule
