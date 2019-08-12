package acer_benchmark;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.text.SimpleDateFormat;
import java.util.Date;
import com.jcraft.jsch.ChannelExec;
import com.jcraft.jsch.JSch;
import com.jcraft.jsch.JSchException;
import com.jcraft.jsch.Session;
import com.jcraft.jsch.UserInfo;

public class RemoteShell {

    private final String username;
    private final String host;
    private final int port;
    private final String connectionLabel;

    private Session session;
    private ChannelExec channel;

    /* RemoteShell constructor */
    public RemoteShell(String user, String pass, String ip, int port) {
        username = user;
        host = ip;
        this.port = port;
        
        connectionLabel = username + "@" + host + ":" + this.port;
        
        // open new connection with specified parameters
        openConnection(pass);
    }
    
    /* returns the connection status of this session */
    public boolean isConnected() {
        return session.isConnected();
    }

    /* returns the respective GPU hostname of the current host */
    private String getGpuHost() {
	if (host.contains("extreme"))
	    return "gpu-a-0.extreme";
	else
	    return "gpu-3-0.saber";
    }

    /* returns the respective GPU loader file of the current host */
    private String getGpuLoader() {
	if (host.contains("extreme"))
	    return "z00_lmod.sh";
	else
	    return "modules.sh";
    }
    
    /* attempt to establish a connection with the host's shell */
    private void openConnection(String password) {
        
        try {
            
            JSch jsch = new JSch();
        
            session = jsch.getSession(username, host, port);
            
	    UserInfo ui = new DuoUserInfo(password);
	    session.setUserInfo(ui);
            //session.setConfig("StrictHostKeyChecking", "no");
            session.setPassword(password);
            session.connect();
            
            System.out.println("> Connected successfully via 2FA to " + connectionLabel); 
        }
        catch (Exception e) {
            System.out.println("> Error trying to connect:\n\t" + e);
        }
        
    }
    
    /* closes the established connection */
    public void closeConnection() {
        
        try {
            
            // disconnect session
            session.disconnect();
            
            System.out.println("> Disconnected successfully from " + connectionLabel);
        }
        catch (Exception e) {
            System.out.println("> Error trying to disconnect:\n\t" + e);
        }
    }
    
    /* runs the benchmark and returns the time result while logging process */
    public void runBenchmark(String type, String info[]) throws JSchException, IOException {
        
        // get directory of benchmark
        String directory = "cd ~/benchmarks/" + info[0] + "/; ";
        
        // get benchmark executable + parameters and send output to text file
        String benchmark = "./" + type + ".out " + info[1] + info[2] + " > " + type + ".txt; ";
        
        // get output of benchmark run
        String output = "less " + type + ".txt >&2;";
        
        // combine above parts into one command
        String command = directory + benchmark + output;
        
        // if running cuda...
        if (type.equals("cuda")) {
            
            // load cuda module before running cuda
	    String load_cuda = "source /etc/profile.d/" + getGpuLoader() + "; source ./load.sh; " + command;
            
            // log into gpu node and then load above commands
            String load_gpu = "ssh " + getGpuHost() + " \""+ load_cuda +"\";";
            
            // set command to include loading gpu and cuda
            command = load_gpu;
        }
        
	// run the benchmark on the remote shell
        runCommand(type, info[0], info[1].substring(info[1].indexOf(' ')).trim(), info[2].substring(info[2].indexOf(' ')).trim(), command);
    }

    /* runs the specified command on the session */
    private void runCommand(String type, String benchmark, String parameter, String threads, String command) throws JSchException, IOException {
        
        // open the exec channel
        channel = (ChannelExec) session.openChannel("exec");
        
        // ready an extended data input stream
        InputStream in = channel.getExtInputStream();
        
        // set the command
        channel.setCommand(command);
        
        System.out.println("> Ran '" + command + "' on " + connectionLabel);
        
        // run the command
        channel.connect();
        
        // read from input stream
        BufferedReader reader = new BufferedReader(new InputStreamReader(in));

        // for each observed input...
        String line;
        while ((line = reader.readLine()) != null) {
            
            // append to log the stream input
            ACER_Benchmark.appendToLog(type, line);
            
            // find output line that shows benchmark result
            if (line.contains("Time =")) {

		// extract time value from current line
                double time = Double.parseDouble(line.substring(line.indexOf('=') + 2, line.indexOf('s')).trim());
                
		// get current date in MM/dd/yyyy format
		String date = new SimpleDateFormat("MM/dd/yyyy").format(new Date());
                
		// store the result
		if (type.equals("cuda"))
		    ACER_Benchmark.writeResult(type, benchmark, time + "," + parameter + "," + threads.substring(3, threads.length()) + ";" + date + "\n");
		else
		    ACER_Benchmark.writeResult(type, benchmark, time + "," + parameter + "," + date + "\n");
            }
        }
        
        channel.disconnect();
    }
    
}
