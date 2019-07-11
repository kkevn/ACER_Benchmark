/******************************************************************************
 * UIC ACER - CPU vs. GPU Benchmarker
 ******************************************************************************
 * By:
 *  Kevin Kowalski  - kkowal28@uic.edu
 ******************************************************************************
 * Description:
 *  todo
 ******************************************************************************/

package acer_benchmark;

import com.jcraft.jsch.JSchException;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.event.EventHandler;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.Alert.AlertType;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Hyperlink;
import javafx.scene.control.ListView;
import javafx.scene.control.PasswordField;
import javafx.scene.control.RadioButton;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.control.ToggleGroup;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.KeyCode;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.FlowPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.FontPosture;
import javafx.scene.text.Text;
import javafx.stage.Stage;
import javafx.stage.Window;
import javafx.stage.WindowEvent;

public class ACER_Benchmark extends Application {

    /* useful variables */
    private int default_port = 22;
    private int port = 22;
    private String default_ip = "systems3.cs.uic.edu";
    private String host = "";
    private String username = "kkowalsk";
    private String password = "";
    private boolean connected = false;
    private RemoteShell remote_shell;
    private String[][] tips = {{"Log in","The first page is the log in."}, 
                               {"Benchmarking","The next page is running benchmarks."},
                               {"Results","The last page is viewing results."}};
    
    /* universal elements */
    private static Stage mainStage;
    private Scene s_connect, s_benchmark, s_results;
    private Font font, t_font, tf_font;

    /* login screen components */
    private ImageView iv_title;
    private Text t_username, t_password, t_port, t_ip;
    private TextField tf_username, tf_port, tf_ip;
    private PasswordField pf_password;
    private Button b_connect;
    private Hyperlink hl_about, hl_contact, hl_help;
    private HBox hb_title, hb_ip_port, hb_connect, hb_footer;
    private VBox vb_title, vb_username, vb_password, vb_ip, vb_port, vb_inputs, vb_connect;
    
    /* benchmark screen components */
    private ImageView iv_title2;
    private BorderPane bp_benchmark;
    private Text t_benchmarks, t_time_c, t_time_cuda;
    private static TextArea ta_log;
    private ToggleGroup tg_benchmarks;
    private RadioButton rb_matrix, rb_test2, rb_test3, rb_test4;
    private ComboBox cb_size, cb_run_count;
    private Button b_run_c, b_run_cuda, b_results, b_logout;
    private Hyperlink hl_about2, hl_contact2, hl_help2;
    private HBox hb_title2, hb_benchmarks, hb_c_cuda, hb_results_logout, hb_footer2;
    private VBox vb_title2, vb_benchmarks, vb_center_benchmarks, vb_c, vb_cuda;

    /* results screen components */
    private BorderPane bp_layout;
    private HBox hb_rounds;
    private VBox vb_center_results;
    private Text t_player, t_word, t_rounds;
    private TextField tf_word;
    private ListView<FlowPane> lv_scores;

    /* creates the scene for the login screen */
    private Parent createConnectScene() {

        // title logo
        iv_title = new ImageView(new Image("/images/OTH.ACER.LT.SM.BLK-500-new.png"));
        iv_title.setFitHeight(64);
        iv_title.setPreserveRatio(true);
        
        // layout for title text components
        hb_title = new HBox();
        hb_title.getChildren().add(iv_title);
        hb_title.setAlignment(Pos.CENTER_LEFT);
        vb_title = new VBox();
        vb_title.getChildren().addAll(hb_title);
        vb_title.setMargin(hb_title, new Insets(0, 0, 0, 32));
        
        // ip field
        t_ip = new Text("Server ");
        t_ip.setFill(Color.BLACK);
        t_ip.setFont(t_font);
        t_ip.setStyle("-fx-font-weight: bold");
        tf_ip = new TextField(default_ip);
        tf_ip.setFont(tf_font);
        tf_ip.setMinWidth(354);
        vb_ip = new VBox();
        vb_ip.getChildren().addAll(t_ip, tf_ip);
        vb_ip.setAlignment(Pos.CENTER_LEFT);

        // port field
        t_port = new Text("Port ");
        t_port.setFill(Color.BLACK);
        t_port.setFont(t_font);
        t_port.setStyle("-fx-font-weight: bold");
        tf_port = new TextField("" + default_port);
        tf_port.setFont(tf_font);
        tf_port.setMaxWidth(64);
        vb_port = new VBox();
        vb_port.getChildren().addAll(t_port, tf_port);
        vb_port.setAlignment(Pos.CENTER_LEFT);
        
        // layout for ip and port components
        hb_ip_port = new HBox();
        hb_ip_port.getChildren().addAll(vb_ip, vb_port);
        hb_ip_port.setAlignment(Pos.CENTER_LEFT);
        hb_ip_port.setMargin(vb_ip, new Insets(0, 16, 0, 0));
        hb_ip_port.setMargin(vb_port, new Insets(0, 0, 0, 16));
        
        // username field
        t_username = new Text("Username ");
        t_username.setFill(Color.BLACK);
        t_username.setFont(t_font);
        t_username.setStyle("-fx-font-weight: bold");
        tf_username = new TextField(username);
        tf_username.setFont(tf_font);
        tf_username.setMaxWidth(180);
        vb_username = new VBox();
        vb_username.getChildren().addAll(t_username, tf_username);
        vb_username.setAlignment(Pos.CENTER_LEFT);
        
        // password field
        t_password = new Text("Password ");
        t_password.setFill(Color.BLACK);
        t_password.setFont(t_font);
        t_password.setStyle("-fx-font-weight: bold");
        pf_password = new PasswordField();
        pf_password.setText(password);
        pf_password.setFont(tf_font);
        pf_password.setMaxWidth(180);
        vb_password = new VBox();
        vb_password.getChildren().addAll(t_password, pf_password);
        vb_password.setAlignment(Pos.CENTER_LEFT);
        
        // layout for inputs
        vb_inputs = new VBox();
        vb_inputs.getChildren().addAll(hb_ip_port, vb_username, vb_password);
        vb_inputs.setMaxWidth(450);
        //vb_inputs.setBackground(new Background(new BackgroundFill(Color.rgb(0, 0, 0, 0.25), new CornerRadii(4), new Insets(-8))));
        vb_inputs.setMargin(hb_ip_port, new Insets(10, 0, 16, 0));
        vb_inputs.setMargin(vb_username, new Insets(0, 0, 16, 0));
        vb_inputs.setMargin(vb_password, new Insets(0, 0, 16, 0));
        
        // connect button
        b_connect = new Button("Connect");
        b_connect.setFont(font);
        b_connect.setAlignment(Pos.CENTER);
        b_connect.setMinHeight(32);
        b_connect.setMinWidth(450);
        b_connect.setPadding(new Insets(0));
        resetButtonStyle(b_connect, "#d50032");
        b_connect.setOnMouseEntered(e -> setButtonHoverStyle(b_connect, "#a00026"));
        b_connect.setOnMousePressed(e -> setButtonPressedStyle(b_connect, "#a00026"));
        b_connect.setOnMouseExited(e -> resetButtonStyle(b_connect, "#d50032"));
        b_connect.setOnAction(e -> connect());
        b_connect.setOnKeyReleased(e -> {
            if (e.getCode() == KeyCode.ENTER){
               connect();
            }
        });
        //b_connect.setDisable(true);
        
        // layout for button row
        hb_connect = new HBox();
        hb_connect.getChildren().addAll(b_connect);
        hb_connect.setAlignment(Pos.CENTER);
        hb_connect.setMargin(vb_ip, new Insets(0, 16, 0, 0));
        
        // about hyperlink
        hl_about = new Hyperlink("About");
        hl_about.setOnAction(e -> new PageDialog(0, "About", tips));
        hl_about.setStyle("-fx-text-fill: #999999;-fx-border-color: transparent;");
        
        // contact hyperlink
        hl_contact = new Hyperlink("Contact");
        hl_contact.setOnAction(e -> showDialogPrompt(AlertType.INFORMATION, "Contact", "Contact", "Email: \tkkowal28@uic.edu\n\nUIC ACER (C) 2019", ButtonType.OK));
        hl_contact.setStyle("-fx-text-fill: #999999;-fx-border-color: transparent;");
        
        // help hyperlink
        hl_help = new Hyperlink("Help");
        hl_help.setOnAction(e -> showDialogPrompt(AlertType.INFORMATION, "Help", "Help", "This is a help dialog box.", ButtonType.OK));
        hl_help.setStyle("-fx-text-fill: #999999;-fx-border-color: transparent;");
        
        // layout for footer
        hb_footer = new HBox();
        hb_footer.getChildren().addAll(hl_about, hl_contact, hl_help);
        hb_footer.setAlignment(Pos.CENTER);
        hb_footer.setMinHeight(32);
        hb_footer.setStyle("-fx-background-color: #252525");
        
        // layout for connect screen
        vb_connect = new VBox();
        vb_connect.getChildren().addAll(vb_title, vb_inputs, hb_connect, hb_footer);
        vb_connect.setAlignment(Pos.CENTER);
        vb_connect.setMinSize(600, 400);
        vb_connect.setMargin(vb_title, new Insets(0, 0, 16, 0));
        vb_connect.setMargin(vb_inputs, new Insets(0, 0, 16, 0));
        vb_connect.setMargin(hb_connect, new Insets(0, 0, 16, 0));
        vb_connect.setMargin(hb_footer, new Insets(0, 0, -32, 0));
        vb_connect.setStyle("-fx-background-color: #F8F8F8;");

        // set the layout to the scene as 600px by 400px
        //s_connect = new Scene(vb_connect, 600, 400);
        return vb_connect;
    }

    /* creates the scene for running benchmarks */
    private void createBenchmarkScene() {
        
        // title logo2
        iv_title2 = new ImageView(new Image("/images/OTH.ACER.LT.SM.BLK-500-new.png"));
        iv_title2.setFitHeight(64);
        iv_title2.setPreserveRatio(true);
        
        // layout for title text components2
        hb_title2 = new HBox();
        hb_title2.getChildren().add(iv_title2);
        hb_title2.setAlignment(Pos.CENTER_LEFT);
        vb_title2 = new VBox();
        vb_title2.getChildren().addAll(hb_title2);
        vb_title2.setMargin(hb_title2, new Insets(0, 0, 0, 32));
        
        // benchmark title field
        t_benchmarks = new Text("BENCHMARKS");
        t_benchmarks.setFill(Color.rgb(0, 30, 98));
        t_benchmarks.setFont(t_font);
        t_benchmarks.setStyle("-fx-border-style: hidden hidden solid hidden;"
                         + "-fx-border-width: 2;"
                         + "-fx-border-color: #d8d8d8;"
                         + "-fx-font-weight: bold;"
                         + "-fx-font-size: 18px");
        
        hb_benchmarks = new HBox();
        hb_benchmarks.getChildren().add(t_benchmarks);
        hb_benchmarks.setStyle("-fx-border-style: hidden hidden solid hidden;"
                         + "-fx-border-width: 2;"
                         + "-fx-padding: 0 48 8 0;"
                         + "-fx-border-color: #d8d8d8;");
        
        // benchmark button components
        tg_benchmarks = new ToggleGroup();
        rb_matrix = new RadioButton("Matrix Multiplication");
        rb_matrix.setToggleGroup(tg_benchmarks);
        rb_test2 = new RadioButton("#test_2\t\t\t");
        rb_test2.setToggleGroup(tg_benchmarks);
        rb_test3 = new RadioButton("#test_3\t\t\t");
        rb_test3.setToggleGroup(tg_benchmarks);
        rb_test4 = new RadioButton("#test_4\t\t\t");
        rb_test4.setToggleGroup(tg_benchmarks);
        tg_benchmarks.selectToggle(rb_matrix);
        setRadioButtonStyle(rb_matrix, rb_test2, rb_test3, rb_test4);
        
        // layout for benchmark list
        vb_benchmarks = new VBox();
        vb_benchmarks.getChildren().addAll(hb_benchmarks,
                rb_matrix, rb_test2, rb_test3, rb_test4);
        //vb_benchmarks.setMaxWidth(450);
        vb_benchmarks.setAlignment(Pos.TOP_LEFT);
        vb_benchmarks.setMargin(hb_benchmarks, new Insets(16));
        vb_benchmarks.setMargin(rb_matrix, new Insets(16));
        vb_benchmarks.setMargin(rb_test2, new Insets(16));
        vb_benchmarks.setMargin(rb_test3, new Insets(16));
        vb_benchmarks.setMargin(rb_test4, new Insets(16));
        
        // text area log field
        ta_log = new TextArea("[" + username + "@" + host + "] > ");
        ta_log.setFont(tf_font);
        ta_log.setMinHeight(256);
        ta_log.setEditable(false);
        ta_log.setStyle("-fx-text-fill: #ffffff;"
                      + "-fx-highlight-fill: #999999;"
                      + "-fx-display-caret: true;"
                      + "-fx-control-inner-background: #000000;");
        
        // c/c++ timer text field
        t_time_c = new Text("--:--");
        //t_time_c.setFill(Color.BLACK);
        t_time_c.setFont(t_font);
        t_time_c.setStyle("-fx-font-size: 32px;");
        //t_time_c.setStyle("-fx-font-weight: bold");
        
        // cuda timer text field
        t_time_cuda = new Text("--:--");
        //t_time_cuda.setFill(Color.BLACK);
        t_time_cuda.setFont(t_font);
        t_time_cuda.setStyle("-fx-font-size: 32px;");
        //t_time_cuda.setStyle("-fx-font-weight: bold");
        
        // c/c++ run button
        b_run_c = new Button("Run C/C++");
        b_run_c.setFont(font);
        b_run_c.setAlignment(Pos.CENTER);
        b_run_c.setMinHeight(32);
        b_run_c.setMinWidth(150);
        b_run_c.setPadding(new Insets(0));
        resetButtonStyle(b_run_c, "#007fa5");
        b_run_c.setOnMouseEntered(e -> setButtonHoverStyle(b_run_c, "#005f7c"));
        b_run_c.setOnMousePressed(e -> setButtonPressedStyle(b_run_c, "#005f7c"));
        b_run_c.setOnMouseExited(e -> resetButtonStyle(b_run_c, "#007fa5"));
        b_run_c.setOnAction(e -> runBenchmark("c/c++"));
        //b_run_c.setDisable(true);
        
        // cuda run button
        b_run_cuda = new Button("Run Cuda");
        b_run_cuda.setFont(font);
        b_run_cuda.setAlignment(Pos.CENTER);
        b_run_cuda.setMinHeight(32);
        b_run_cuda.setMinWidth(150);
        b_run_cuda.setPadding(new Insets(0));
        resetButtonStyle(b_run_cuda, "#007fa5");
        b_run_cuda.setOnMouseEntered(e -> setButtonHoverStyle(b_run_cuda, "#005f7c"));
        b_run_cuda.setOnMousePressed(e -> setButtonPressedStyle(b_run_cuda, "#005f7c"));
        b_run_cuda.setOnMouseExited(e -> resetButtonStyle(b_run_cuda, "#007fa5"));
        b_run_cuda.setOnAction(e -> runBenchmark("cuda"));
        //b_run_cuda.setDisable(true);
        
        // layout for c/c++ fields
        vb_c = new VBox();
        vb_c.getChildren().addAll(t_time_c, b_run_c);
        //vb_c.setMaxWidth(450);
        vb_c.setAlignment(Pos.CENTER);
        vb_c.setMargin(t_time_c, new Insets(0, 72, 16, 0));
        vb_c.setMargin(b_run_c, new Insets(0, 72, 0, 0));
        
        // layout for cuda fields
        vb_cuda = new VBox();
        vb_cuda.getChildren().addAll(t_time_cuda, b_run_cuda);
        //vb_cuda.setMaxWidth(450);
        vb_cuda.setAlignment(Pos.CENTER);
        vb_cuda.setMargin(t_time_cuda, new Insets(0, 0, 16, 72));
        vb_cuda.setMargin(b_run_cuda, new Insets(0, 0, 0, 72));
        
        // layout for c vs cuda row
        hb_c_cuda = new HBox();
        hb_c_cuda.getChildren().addAll(vb_c, vb_cuda);
        hb_c_cuda.setAlignment(Pos.CENTER);
        //hb_c_cuda.setMargin(vb_ip, new Insets(0, 16, 0, 0));
        
        // results button
        b_results = new Button("Results");
        b_results.setFont(font);
        b_results.setAlignment(Pos.CENTER);
        b_results.setMinHeight(32);
        b_results.setMinWidth(216);
        b_results.setPadding(new Insets(0));
        resetButtonStyle(b_results, "#d50032");
        b_results.setOnMouseEntered(e -> setButtonHoverStyle(b_results, "#a00026"));
        b_results.setOnMousePressed(e -> setButtonPressedStyle(b_results, "#a00026"));
        b_results.setOnMouseExited(e -> resetButtonStyle(b_results, "#d50032"));
        //b_results.setOnAction(e -> setCurrentScene(2));
        //b_results.setDisable(true);
        
        // logout button
        b_logout = new Button("Logout");
        b_logout.setFont(font);
        b_logout.setAlignment(Pos.CENTER);
        b_logout.setMinHeight(32);
        b_logout.setMinWidth(216);
        b_logout.setPadding(new Insets(0));
        resetButtonStyle(b_logout, "#d50032");
        b_logout.setOnMouseEntered(e -> setButtonHoverStyle(b_logout, "#a00026"));
        b_logout.setOnMousePressed(e -> setButtonPressedStyle(b_logout, "#a00026"));
        b_logout.setOnMouseExited(e -> resetButtonStyle(b_logout, "#d50032"));
        b_logout.setOnAction(e -> disconnect());
        //b_logout.setDisable(true);
        
        // layout for results and logout row
        hb_results_logout = new HBox();
        hb_results_logout.getChildren().addAll(b_results, b_logout);
        hb_results_logout.setAlignment(Pos.CENTER);
        hb_results_logout.setMargin(b_results, new Insets(0, 32, 0, 0));
        hb_results_logout.setMargin(b_logout, new Insets(0, 0, 0, 32));
        
        // benchmark controls layout in center
        vb_center_benchmarks = new VBox();
        vb_center_benchmarks.getChildren().addAll(ta_log, hb_c_cuda, hb_results_logout);
        vb_center_benchmarks.setAlignment(Pos.TOP_CENTER);
        vb_cuda.setMargin(ta_log, new Insets(16, 0, 16, 0));
        vb_cuda.setMargin(hb_c_cuda, new Insets(0, 0, 32, 0));
        vb_cuda.setMargin(hb_results_logout, new Insets(0, 0, 0, 0));
        vb_center_benchmarks.setStyle("-fx-background-color: #F8F8F8;");
        
        // about hyperlink
        hl_about2 = new Hyperlink("About");
        hl_about2.setOnAction(e -> new PageDialog(0, "About", tips));
        hl_about2.setStyle("-fx-text-fill: #999999;-fx-border-color: transparent;");
        
        // contact hyperlink
        hl_contact2 = new Hyperlink("Contact");
        hl_contact2.setOnAction(e -> showDialogPrompt(AlertType.INFORMATION, "Contact", "Contact", "Email: \tkkowal28@uic.edu\n\nUIC ACER (C) 2019", ButtonType.OK));
        hl_contact2.setStyle("-fx-text-fill: #999999;-fx-border-color: transparent;");
        
        // help hyperlink
        hl_help2 = new Hyperlink("Help");
        hl_help2.setOnAction(e -> showDialogPrompt(AlertType.INFORMATION, "Help", "Help", "This is a help dialog box.", ButtonType.OK));
        hl_help2.setStyle("-fx-text-fill: #999999;-fx-border-color: transparent;");
        
        // layout for footer2
        hb_footer2 = new HBox();
        hb_footer2.getChildren().addAll(hl_about2, hl_contact2, hl_help2);
        hb_footer2.setAlignment(Pos.CENTER);
        hb_footer2.setMinHeight(32);
        hb_footer2.setStyle("-fx-background-color: #252525");
        
        // layout for benchmark screen
        bp_benchmark = new BorderPane();
        bp_benchmark.setTop(vb_title2);
        bp_benchmark.setLeft(vb_benchmarks);
        bp_benchmark.setCenter(vb_center_benchmarks);
        bp_benchmark.setRight(new Text("parameters"));
        bp_benchmark.setBottom(hb_footer2);
        bp_benchmark.setMargin(vb_title2, new Insets(16, 0, 16, 0));
        //bp_benchmark.setMargin(hb_footer2, new Insets(0, 0, -80, 0));
        bp_benchmark.setStyle("-fx-background-color: #F8F8F8;");

        // set the layout to the scene as 800px by 600px
        s_benchmark = new Scene(bp_benchmark, 800, 600);

        //return bp_benchmark;
    }
    
    /* creates the scene for viewing benchmark results */
    private void createResultsScene() {

        

        // set the layout to the scene as 800px by 600px
        //s_results = new Scene(bp_layout, 800, 600);

        //return bp_results;
    }

    public static void main(String[] args) {
        launch(args);
    }

    public void start(Stage primaryStage) throws Exception {

        // store refernce to primaryStage
        mainStage = primaryStage;

        // initialize UI fonts
        font = Font.font("Monospace", FontPosture.REGULAR, 32);
        t_font = Font.font("Arial", FontPosture.REGULAR, 14);
        tf_font = Font.font("Monospace", FontPosture.REGULAR, 14);

        // load all three scenes and jump to connect scene
        s_connect = new Scene(createConnectScene());
        createBenchmarkScene();
        createResultsScene();
        mainStage.setTitle("ACER Benchmark - Connect to HPC Servers");
        mainStage.setScene(s_connect);
        //mainStage.setScene(s_benchmark);
        //mainStage.setScene(s_results);
        mainStage.setResizable(false);
        mainStage.setOnCloseRequest(e -> safetyExit());
        mainStage.show();
        b_connect.requestFocus();
    }
    
    /* called when application closed, ensures all connections terminated before exiting */
    private void safetyExit() {
        
        // if active SSH connection opened, error here
        if (connected) {
            
            // close the opened connection(s)
            //remote_shell.closeConnection();
            disconnect();
        }
        
        Platform.exit();
    }
    
    /* connects to a server on valid text field inputs */
    private void connect() {

        try {

            // get username, IP, and port from text fields
            username = tf_username.getText();
            host = tf_ip.getText();
            port = Integer.parseInt(tf_port.getText());
            
            // create the new connection
            remote_shell = new RemoteShell(username, pf_password.getText(), host, port);
            
            // test if successful connection
            if (remote_shell.isConnected()) {
                
                // update connected status
                connected = true;
                
                // set to benchmark scene
                setCurrentScene(1);
            }
            
            // otherwise prompt error message
            else {
                showDialogPrompt(AlertType.ERROR, "Failed Login", "Failed Login", "Ensure all login credentials are accurate.", ButtonType.OK);
            }
            
            
        } catch (Exception e) {
            System.out.println("> Exception @ connect()\n" + e);
            //showDialogPrompt(AlertType.ERROR, "Failed Login", "Failed Login", "Ensure all login credentials are accurate.", ButtonType.OK);
        }
    }
    
    /* disconnect from server and return to login screen */
    private void disconnect() {

        try {
            
            // reset default login inputs
            tf_username.setText(username);
            pf_password.clear();
            tf_ip.setText(host);
            tf_port.setText("" + port);
            
            
            // disconnect only if connected
            if (connected) {
                
                // close the existing connection
                remote_shell.closeConnection();

                // update connected status
                connected = false;

                // set to login scene
                setCurrentScene(0);
                
                // set focus to password field
                pf_password.requestFocus();
            }
            else {
                System.out.println("> No connections to disconnect from");
            }
            
        } catch (Exception e) {
            System.out.println("> Failed to disconnect from server");
        }
    }

    /* sets the GUI to the specified mode */
    public void setCurrentScene(int mode) {

        // title variable for each scene
        String title = "ACER Benchmark - ";

        // set scene according to specified parameter
        switch (mode) {

            // mode 0 -> set to login screen
            case 0:
                title += "Connect to HPC Servers";
                mainStage.setScene(s_connect);
                tf_ip.requestFocus();
                break;

            // mode 1 -> set to benchmark screen
            case 1:
                title += "Run Benchmarks";
                mainStage.setScene(s_benchmark);
                break;
                
            // mode 2 -> set to results screen
            case 2:
                title += "Benchmark Results";
                //bp_benchmark.setCenter(graph);
                //mainStage.setScene(s_results);
                break;
        }

        // set scene title, prevent resizing of window, and show the scene
        mainStage.setTitle(title);
        mainStage.setResizable(false);
        mainStage.show();
    }
    
    /* resets the specified button's style */
    private void resetButtonStyle(Button b, String hex_color) {
        b.setStyle("-fx-background-color: " + hex_color + ";"
                + "-fx-text-fill: black;"
                + "-fx-font-weight: bold;"
                + "-fx-font-family: Calibri;"
                + "-fx-font-size: "+ t_font.getSize() + "px;");
    }
    
    /* adjusts the specified button's style for hover event */
    private void setButtonHoverStyle(Button b, String hex_color) {
        b.setStyle("-fx-background-color: " + hex_color + ";"
                + "-fx-text-fill: black;"
                + "-fx-font-weight: bold;"
                + "-fx-font-family: Calibri;"
                + "-fx-font-size: "+ t_font.getSize() + "px;");
    }
    
    /* adjusts the specified button's style for press event */
    private void setButtonPressedStyle(Button b, String hex_color) {
        b.setStyle("-fx-background-color: " + hex_color + ";"
                + "-fx-text-fill: black;"
                + "-fx-font-weight: bold;"
                + "-fx-font-family: Calibri;"
                + "-fx-font-size: "+ t_font.getSize() + "px;");
    }
    
    /* sets the style to the specified radio button */
    private void setRadioButtonStyle(RadioButton... rb_list) {
        for (RadioButton rb : rb_list) {
            rb.setStyle("-fx-border-style: hidden hidden solid hidden;"
                  + "-fx-border-width: 2;"
                  + "-fx-padding: -16 48 16 0;"
                  + "-fx-border-color: #d8d8d8;");
        }
    }
    
    /* wrapper for showDialogPrompt() */
    public static void invokePrompt(Alert.AlertType at, String title, String header, String message, ButtonType ...bt) {
        showDialogPrompt(at, title, header, message, bt);
    }
    
    /* display a dialog box with specified parameters */
    private static void showDialogPrompt(Alert.AlertType at, String title, String header, String message, ButtonType ...bt) {

        // spawn a dialog box (on main JavaFX thread)
        Platform.runLater(new Runnable() {
            @Override
            public void run() {

                // spawn the dialog box
                Alert alert = new Alert(at, message, bt);
                alert.setTitle(title);
                alert.setHeaderText(header);
                //alert.setResizable(false);
                
                if (title.equals("About")) {
                    alert.setWidth(400);
                    alert.setHeight(300);
                }
                
                // set dialog box position relative to main program's position on desktop
                final Window window = alert.getDialogPane().getScene().getWindow();
                window.addEventHandler(WindowEvent.WINDOW_SHOWN, new EventHandler<WindowEvent>() {
                    @Override
                    public void handle(WindowEvent event) {
                        window.setX((mainStage.getX() + mainStage.getWidth() / 2) - (window.getWidth() / 2));
                        window.setY((mainStage.getY() + mainStage.getHeight() / 2) - (window.getHeight() / 2));
                    }
                });
                
                alert.showAndWait();

                // user clicked OK
                if (alert.getResult() == ButtonType.OK) {
                    
                    // depending on current prompt perform some action
                    switch (title) {
                        case "Help":
                        case "Failed Login":
                            // do nothing
                            break;
                    }
                }
            }
        });
    }
    
    /* wrapper for appending to text area log */
    public static void appendToLog(String s) {
        ta_log.appendText(s + "\n");
    }
    
    /* runs the specified benchmark */
    private void runBenchmark(String type) {
        
        // get selected benchmark as ID
        RadioButton rb = (RadioButton)tg_benchmarks.getSelectedToggle();
        String benchmark = "";
        switch (rb.getText().trim()) {
            case "Matrix Multiplication":
                benchmark = "matrix_mult";
                break;
            case "#test_2":
                benchmark = "t2";
                break;
            case "#test_3":
                benchmark = "t3";
                break;
            case "#test_4":
                benchmark = "t4";
                break;
        }
        
        
        // if C/C++ benchmark button was pressed
        if (type.toLowerCase().contains("c/c++")) {
            try {
                t_time_c.setText("" + remote_shell.runBenchmark("c/c++", benchmark) + "ms");
            } catch (JSchException ex) {
                Logger.getLogger(ACER_Benchmark.class.getName()).log(Level.SEVERE, null, ex);
                System.out.println("> Error @ runBenchmark()-1");
            } catch (IOException ex) {
                Logger.getLogger(ACER_Benchmark.class.getName()).log(Level.SEVERE, null, ex);
                System.out.println("> Error @ runBenchmark()-2");
            }
        }
        
        // otherwise Cuda benchmark button was pressed
        else {
            try {
                t_time_cuda.setText("" + remote_shell.runBenchmark("cuda", benchmark) + "ms");
            } catch (JSchException ex) {
                Logger.getLogger(ACER_Benchmark.class.getName()).log(Level.SEVERE, null, ex);
                System.out.println("> Error @ runBenchmark()-3");
            } catch (IOException ex) {
                Logger.getLogger(ACER_Benchmark.class.getName()).log(Level.SEVERE, null, ex);
                System.out.println("> Error @ runBenchmark()-4");
            }
        }
        
    }
}
