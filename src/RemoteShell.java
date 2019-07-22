package acer_benchmark;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import com.jcraft.jsch.ChannelExec;
import com.jcraft.jsch.JSch;
import com.jcraft.jsch.JSchException;
import com.jcraft.jsch.Session;
import java.io.IOException;

public class RemoteShell {

    private String username;
    private String password;
    private String host;
    private int port;
    private String connectionLabel;

    private Session session;
    private ChannelExec channel;
    
    private List<String> outputs = new ArrayList<>();

    /* RemoteShell constructor */
    public RemoteShell(String user, String pass, String ip, int port) {
        username = user;
        password = pass;
        host = ip;
        this.port = port;
        
        connectionLabel = username + "@" + host + ":" + this.port;
        
        // close existing connection if one exists, error here
        /*if (isConnected()) {
            closeConnection();
        }*/
        
        // open new connection with specified parameters
        openConnection();
    }
    
    /* return the list of observed shell outputs */
    public List<String> getOutputs() {
        return outputs;
    }
    
    /* returns the connection status of this session */
    public boolean isConnected() {
        return session.isConnected();
    }
    
    /* runs the benchmark and returns the time result while logging process */
    public String runBenchmark(String type, String benchmark, int size) throws JSchException, IOException {
        
        double time;
        String result = "";
        
        String command = "cd " + benchmark + "; time ./" + type;
        
        // set command as C/C++ benchmark
        if (type.equals("c_cpp")) {
            command = "cd acer_test; time ./a.out";
        }
        
        // saxpy
        // time for i in {1..1000}; do clear; ./a.out 1000000 1; done
        
        // set command as Cuda benchmark
        else {
            command = "cd acer_test; time ./a.out";
        }
        
        
        ACER_Benchmark.appendToLog("> Initiating " + type.toUpperCase() + " " + benchmark + " benchmark...");
        
        runCommand(command);
        
        for (String line : outputs) {
            
            if ((line.toLowerCase()).contains("time")) {
                time = Double.parseDouble(line.substring(line.indexOf(':') + 2, line.length()).trim());
                        
                result = time + "," + size;
                //ACER_Benchmark.writeResult(type, benchmark, result);
            }
        }
        
        return result;
    }
    
    
    /* attempt to establish a connection with the host's shell */
    private void openConnection() {
        
        try {
            
            JSch jsch = new JSch();
        
            session = jsch.getSession(username, host, port);
            
            session.setConfig("StrictHostKeyChecking", "no");
            session.setPassword(password);
            session.connect();
            System.out.println("> Connected successfully to " + connectionLabel);
            
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
    
    /* runs the specified command on the session */
    private void runCommand(String command) throws JSchException, IOException {
        
        // open the exec channel
        channel = (ChannelExec) session.openChannel("exec");
        
        // ready an input stream
        InputStream in = channel.getInputStream();
        
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
            ACER_Benchmark.appendToLog(line);
            
            // store observed input
            outputs.add(line);
        }
        
        // close the exec channel
        channel.disconnect();
        
        // notify GUI that benchmark is complete
        ACER_Benchmark.appendToLog("> Ready!");
    }
    
}
