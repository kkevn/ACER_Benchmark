/******************************************************************************
 * UIC ACER - C/C++ vs. Cuda Benchmarker
 ******************************************************************************
 * By:
 *  Kevin Kowalski - kkowal28@uic.edu
 ******************************************************************************
 * Description:
 *  Simple tool for observing performance differences in C/C++ vs. Cuda
 *  - Remote into Saber GPU cluster
 *  - Run both C/C++ and Cuda versions of any available benchmark (stored 
 *    and compiled on Saber)
 *  - View results in a meaningful way
 ******************************************************************************/

package acer_benchmark;

import com.jcraft.jsch.JSchException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.chart.XYChart.Data;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Hyperlink;
import javafx.scene.control.PasswordField;
import javafx.scene.control.RadioButton;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.control.ToggleGroup;
import javafx.scene.control.Tooltip;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.KeyCode;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.FontPosture;
import javafx.scene.text.Text;
import javafx.stage.Stage;

public class ACER_Benchmark extends Application {

    /* useful variables */
    private int default_port = 22;
    private int port = 22;
    private String default_ip = "login-3.extreme.acer.uic.edu";
    private String login1 = "login-1.saber.acer.uic.edu";
    private String login2 = "login-2.saber.acer.uic.edu";
    private String login3 = "login-3.extreme.acer.uic.edu";
    private String host = "";
    private String username = "kkowal28";
    private String password = "";
    private boolean connected = false;
    private boolean inResultsView = false;
    private static boolean hasNotification = false;
    private RemoteShell remote_shell;
    private static PageDialog pd_message;
    private String[][] tips = {{"Log in","The first page is the log in."}, 
                               {"Benchmarking","The next page is running benchmarks."},
                               {"Results","The last page is viewing results."}};

    /* universal elements */
    private static Stage mainStage;
    private Scene s_connect, s_benchmark, s_results;
    private Font font, t_font, tf_font;

    /* login screen components */
    private ImageView iv_title;
    private static BorderPane bp_login;
    private Text t_username, t_password, t_port, t_ip;
    private TextField tf_username, tf_port;
    private ComboBox<String> cb_ip;
    private PasswordField pf_password;
    private Button b_connect;
    private Hyperlink hl_about, hl_contact, hl_help;
    private HBox hb_title, hb_ip_port, hb_connect, hb_footer;
    private VBox vb_title, vb_username, vb_password, vb_ip, vb_port, vb_inputs;
    private static VBox vb_connect;
    
    /* benchmark screen components */
    private ImageView iv_title2;
    private BorderPane bp_benchmark;
    private Text t_benchmarks, t_params, t_size, t_runs, t_threads;
    private static TextArea ta_log_c;
    private static TextArea ta_log_cuda;
    private ToggleGroup tg_benchmarks;
    private RadioButton rb_matrix, rb_saxpy, rb_vector, rb_test4;
    private ComboBox<Integer> cb_size, cb_runs, cb_threads;
    private Button b_run_c, b_run_cuda, b_clear, b_results, b_logout;
    private HBox hb_title2, hb_benchmarks, hb_logs_param, hb_params, hb_c_cuda, hb_results_logout;
    private VBox vb_title2, vb_benchmarks, vb_params, vb_size, vb_runs, vb_threads, vb_c, vb_cuda, vb_results_logout;
    private static VBox vb_center_benchmarks;
    private static VBox vb_logs_param;

    /* results screen components */
    private static VBox vb_results;
    private NumberAxis x_time, y_size;
    private LineChart<Number, Number> lc_graph;
    private XYChart.Series xyc_s_c, xyc_s_cuda;

    /* creates the scene for the login screen */
    private Parent createConnectScene() {

        // title logo
        iv_title = new ImageView(new Image("file:src/main/resources/images/OTH.ACER.LT.SM.BLK-500-new.png"));
        iv_title.setFitHeight(96);
        iv_title.setPreserveRatio(true);
        
        // layout for title text components
        hb_title = new HBox();
        hb_title.getChildren().add(iv_title);
        hb_title.setAlignment(Pos.TOP_LEFT);
        vb_title = new VBox();
        vb_title.getChildren().addAll(hb_title);
        vb_title.setMargin(hb_title, new Insets(0, 0, 0, 32));
        
        // ip field
        t_ip = new Text("Server ");
        t_ip.setFill(Color.BLACK);
        t_ip.setFont(t_font);
        t_ip.setStyle("-fx-font-weight: bold");
	cb_ip = new ComboBox<String>();
	cb_ip.getItems().addAll(login1, login2, login3);
	cb_ip.setStyle("-fx-font: 14px \"Monospace\";");
	cb_ip.getSelectionModel().selectFirst();
	cb_ip.setEditable(true);
	cb_ip.setMinWidth(354);
        vb_ip = new VBox();
        vb_ip.getChildren().addAll(t_ip, cb_ip);
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
            if (e.getCode() == KeyCode.ENTER) {
                connect();
            }
        });
        
        // layout for button row
        hb_connect = new HBox();
        hb_connect.getChildren().addAll(b_connect);
        hb_connect.setAlignment(Pos.CENTER);
        //hb_connect.setMargin(vb_ip, new Insets(0, 16, 0, 0));
        
        // about hyperlink
        hl_about = new Hyperlink("About");
        hl_about.setOnAction(e -> drawDialog("About"));
        hl_about.setStyle("-fx-text-fill: #999999;-fx-border-color: transparent;");
        
        // contact hyperlink
        hl_contact = new Hyperlink("Contact");
        hl_contact.setOnAction(e -> drawDialog("Contact"));
        hl_contact.setStyle("-fx-text-fill: #999999;-fx-border-color: transparent;");
        
        // help hyperlink
        hl_help = new Hyperlink("Help");
        hl_help.setOnAction(e -> drawDialog("Help"));
        hl_help.setStyle("-fx-text-fill: #999999;-fx-border-color: transparent;");
        
        // layout for footer
        hb_footer = new HBox();
        hb_footer.getChildren().addAll(hl_about, hl_contact, hl_help);
        hb_footer.setAlignment(Pos.CENTER);
        hb_footer.setMinHeight(32);
        hb_footer.setStyle("-fx-background-color: #252525");
        
        // layout for connect screen
        vb_connect = new VBox();
        vb_connect.getChildren().addAll(vb_inputs, hb_connect);
        vb_connect.setAlignment(Pos.TOP_CENTER);
        vb_connect.setMinSize(600, 400);
        vb_connect.setMargin(vb_inputs, new Insets(196, 0, 16, 0));
        vb_connect.setMargin(hb_connect, new Insets(0, 0, 0, 0));
        vb_connect.setStyle("-fx-background-color: #F8F8F8;");

	// outer layout for connect screen
        bp_login = new BorderPane();
	bp_login.setTop(vb_title);
        bp_login.setCenter(vb_connect);
	bp_login.setBottom(hb_footer);
        bp_login.setMargin(vb_title, new Insets(16, 0, 16, 0));
        bp_login.setStyle("-fx-background-color: #F8F8F8;");

        // set the layout to the scene as 600px by 400px
        //s_connect = new Scene(bp_login, 600, 400);
        return bp_login;
    }

    /* creates the scene for running benchmarks */
    private void createBenchmarkScene() {
        
        // title logo2
        iv_title2 = new ImageView(new Image("file:src/main/resources/images/OTH.ACER.LT.SM.BLK-500-new.png"));
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
        t_benchmarks.setStyle("-fx-font-weight: bold;"
                         + "-fx-font-size: 18px");
        
        hb_benchmarks = new HBox();
        hb_benchmarks.getChildren().add(t_benchmarks);
        hb_benchmarks.setStyle("-fx-border-style: solid;"
                         + "-fx-border-width: 0 0 2 0;"
                         + "-fx-padding: 0 48 8 0;"
                         + "-fx-border-color: #d8d8d8;");
        
        // benchmark button components
        tg_benchmarks = new ToggleGroup();
        rb_matrix = new RadioButton("Matrix Multiplication\t");
        rb_matrix.setToggleGroup(tg_benchmarks);
        rb_matrix.setOnAction(e -> benchmarkButtonListener());
        rb_saxpy = new RadioButton("Single-Precision AX+Y\t");
        rb_saxpy.setToggleGroup(tg_benchmarks);
        rb_saxpy.setOnAction(e -> benchmarkButtonListener());
        rb_vector = new RadioButton("Vector Addition\t\t");
        rb_vector.setToggleGroup(tg_benchmarks);
        rb_vector.setOnAction(e -> benchmarkButtonListener());
        rb_test4 = new RadioButton("#test_4\t\t\t\t");
        rb_test4.setToggleGroup(tg_benchmarks);
        rb_test4.setOnAction(e -> benchmarkButtonListener());
        tg_benchmarks.selectToggle(rb_matrix);
        setRadioButtonStyle(rb_matrix, rb_saxpy, rb_vector, rb_test4);      

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
        b_results.setOnAction(e -> benchmarkSceneSwitch());
        
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

	// layout for results and logout row
        vb_results_logout = new VBox();
        vb_results_logout.getChildren().addAll(b_results, b_logout);
        vb_results_logout.setAlignment(Pos.CENTER_LEFT);
        vb_results_logout.setMargin(b_results, new Insets(0, 0, 16, 0));
        vb_results_logout.setMargin(b_logout, new Insets(0, 0, 0, 0));

        // layout for benchmark list
        vb_benchmarks = new VBox();
        vb_benchmarks.getChildren().addAll(hb_benchmarks, rb_matrix, rb_saxpy, rb_vector, rb_test4, vb_results_logout);
        //vb_benchmarks.setMaxWidth(450);
        vb_benchmarks.setAlignment(Pos.TOP_LEFT);
        vb_benchmarks.setMargin(hb_benchmarks, new Insets(16));
        vb_benchmarks.setMargin(rb_matrix, new Insets(16));
        vb_benchmarks.setMargin(rb_saxpy, new Insets(16));
        vb_benchmarks.setMargin(rb_vector, new Insets(16));
        vb_benchmarks.setMargin(rb_test4, new Insets(16));
	vb_benchmarks.setMargin(vb_results_logout, new Insets(320, 0, 8, 16));
        
        // text area log for c benchmarks
        ta_log_c = new TextArea("Waiting for C/C++ benchmark to be run...\n\n");
        ta_log_c.setFont(tf_font);
        ta_log_c.setMinHeight(384);
	ta_log_c.setMinWidth(640);
        ta_log_c.setEditable(false);
        ta_log_c.setStyle("-fx-text-fill: #ffffff;"
                      + "-fx-highlight-fill: #999999;"
                      + "-fx-display-caret: true;"
                      + "-fx-control-inner-background: #000000;");

	// text area log for cuda benchmarks
        ta_log_cuda = new TextArea("Waiting for CUDA benchmark to be run...\n\n");
        ta_log_cuda.setFont(tf_font);
        ta_log_cuda.setMinHeight(384);
	ta_log_cuda.setMinWidth(640);
        ta_log_cuda.setEditable(false);
        ta_log_cuda.setStyle("-fx-text-fill: #ffffff;"
                      + "-fx-highlight-fill: #999999;"
                      + "-fx-display-caret: true;"
                      + "-fx-control-inner-background: #000000;");
        
	// paremeter label
	t_params = new Text("    Benchmark Parameters    ");
	t_params.setFill(Color.BLACK);
        t_params.setFont(t_font);
        t_params.setStyle("-fx-font-weight: bold;"
		+ "-fx-font-size: 16px;");
        
        hb_params = new HBox();
        hb_params.getChildren().add(t_params);
	hb_params.setAlignment(Pos.CENTER);
        hb_params.setStyle("-fx-border-style: solid;"
                         + "-fx-border-width: 0 0 2 0;"
                         + "-fx-padding: 0 0 8 0;"
                         + "-fx-border-color: #d8d8d8;");

	// size parameter options
	t_size = new Text("Size of N Elements");
	t_size.setFill(Color.BLACK);
        t_size.setFont(t_font);
        t_size.setStyle("-fx-font-weight: bold");
	cb_size = new ComboBox<Integer>();
	cb_size.getItems().addAll(1000, 10000, 100000, 1000000);
	cb_size.getSelectionModel().selectFirst();
	cb_size.setMinWidth(128);
	vb_size = new VBox();
        vb_size.getChildren().addAll(t_size, cb_size);
        vb_size.setAlignment(Pos.CENTER);
	
	// size parameter options
	t_runs = new Text("Number of Runs");
	t_runs.setFill(Color.BLACK);
        t_runs.setFont(t_font);
        t_runs.setStyle("-fx-font-weight: bold");
	cb_runs = new ComboBox<Integer>();
	cb_runs.getItems().addAll(1, 100, 1000, 10000);
	cb_runs.getSelectionModel().selectFirst();
	cb_runs.setMinWidth(128);
	vb_runs = new VBox();
        vb_runs.getChildren().addAll(t_runs, cb_runs);
        vb_runs.setAlignment(Pos.CENTER);
	//cb_runs.setDisable(true);
        
        // size parameter options
	t_threads = new Text("Threads per Block");
	t_threads.setFill(Color.BLACK);
        t_threads.setFont(t_font);
        t_threads.setStyle("-fx-font-weight: bold");
	cb_threads = new ComboBox<Integer>();
	cb_threads.getItems().addAll(1, 32, 64, 128, 256, 512, 1024);
	cb_threads.getSelectionModel().select(4);
	cb_threads.setMinWidth(128);
	vb_threads = new VBox();
        vb_threads.getChildren().addAll(t_threads, cb_threads);
        vb_threads.setAlignment(Pos.CENTER);
	
	// clear logs button
        b_clear = new Button("< Clear Logs >");
        b_clear.setFont(font);
        b_clear.setAlignment(Pos.CENTER);
        b_clear.setMinHeight(32);
        b_clear.setMinWidth(128);
        b_clear.setPadding(new Insets(0));
        resetButtonStyle(b_clear, "#007fa5");
        b_clear.setOnMouseEntered(e -> setButtonHoverStyle(b_clear, "#005f7c"));
        b_clear.setOnMousePressed(e -> setButtonPressedStyle(b_clear, "#005f7c"));
        b_clear.setOnMouseExited(e -> resetButtonStyle(b_clear, "#007fa5"));
        b_clear.setOnAction(e -> {
		ta_log_c.clear();
		ta_log_c.appendText("Waiting for C/C++ benchmark to be run...\n\n");
		ta_log_cuda.clear();
		ta_log_cuda.appendText("Waiting for CUDA benchmark to be run...\n\n");
	});

	// layout for parameters
	vb_params = new VBox();
	vb_params.getChildren().addAll(hb_params, vb_size, vb_runs, vb_threads, b_clear);
	vb_params.setAlignment(Pos.TOP_CENTER);
	vb_params.setMargin(hb_params, new Insets(0, 0, 16, 0));
	vb_params.setMargin(vb_size, new Insets(16, 0, 16, 0));
        vb_params.setMargin(vb_runs, new Insets(16, 0, 16, 0));
        vb_params.setMargin(vb_threads, new Insets(16, 0, 16, 0));
	vb_params.setMargin(b_clear, new Insets(64, 0, 0, 0));
        
        // c/c++ run button
        b_run_c = new Button("Run C/C++");
        b_run_c.setFont(font);
        b_run_c.setAlignment(Pos.CENTER);
        b_run_c.setMinHeight(32);
        b_run_c.setMinWidth(256);
        b_run_c.setPadding(new Insets(0));
        resetButtonStyle(b_run_c, "#007fa5");
        b_run_c.setOnMouseEntered(e -> setButtonHoverStyle(b_run_c, "#005f7c"));
        b_run_c.setOnMousePressed(e -> setButtonPressedStyle(b_run_c, "#005f7c"));
        b_run_c.setOnMouseExited(e -> resetButtonStyle(b_run_c, "#007fa5"));
        b_run_c.setOnAction(e -> runBenchmark("c_cpp"));
        
        // cuda run button
        b_run_cuda = new Button("Run Cuda");
        b_run_cuda.setFont(font);
        b_run_cuda.setAlignment(Pos.CENTER);
        b_run_cuda.setMinHeight(32);
        b_run_cuda.setMinWidth(256);
        b_run_cuda.setPadding(new Insets(0));
        resetButtonStyle(b_run_cuda, "#007fa5");
        b_run_cuda.setOnMouseEntered(e -> setButtonHoverStyle(b_run_cuda, "#005f7c"));
        b_run_cuda.setOnMousePressed(e -> setButtonPressedStyle(b_run_cuda, "#005f7c"));
        b_run_cuda.setOnMouseExited(e -> resetButtonStyle(b_run_cuda, "#007fa5"));
        b_run_cuda.setOnAction(e -> runBenchmark("cuda"));
        
        // layout for c/c++ fields
        vb_c = new VBox();
        vb_c.getChildren().addAll(ta_log_c, b_run_c);
        //vb_c.setMaxWidth(450);
        vb_c.setAlignment(Pos.CENTER);
        //vb_c.setMargin(ta_log_c, new Insets(0, 72, 16, 0));
        vb_c.setMargin(b_run_c, new Insets(32, 0, 0, 0));
        
        // layout for cuda fields
        vb_cuda = new VBox();
        vb_cuda.getChildren().addAll(ta_log_cuda, b_run_cuda);
        //vb_cuda.setMaxWidth(450);
        vb_cuda.setAlignment(Pos.CENTER);
        //vb_cuda.setMargin(ta_log_cuda, new Insets(0, 0, 16, 72));
        vb_cuda.setMargin(b_run_cuda, new Insets(32, 0, 0, 0));
        
        // layout for c vs cuda row
        //hb_c_cuda = new HBox();
        //hb_c_cuda.getChildren().addAll(b_run_c, b_run_cuda);
        //hb_c_cuda.setAlignment(Pos.CENTER_LEFT);
	//hb_c_cuda.setMargin(b_run_c, new Insets(0, 72, 0, 0));
        //hb_c_cuda.setMargin(b_run_cuda, new Insets(0, 0, 0, 72));
        
        
        
        // benchmark controls layout in center
        /*vb_center_benchmarks = new VBox();
        vb_center_benchmarks.getChildren().addAll(hb_logs_param, hb_c_cuda, hb_results_logout);
        vb_center_benchmarks.setAlignment(Pos.TOP_CENTER);
        vb_cuda.setMargin(hb_logs_param, new Insets(16, 0, 16, 0));
        vb_cuda.setMargin(hb_c_cuda, new Insets(0, 0, 32, 0));
        vb_cuda.setMargin(hb_results_logout, new Insets(0, 0, 0, 0));
        vb_center_benchmarks.setStyle("-fx-background-color: #F8F8F8;");
        */
	// layout for logs and parameter list
	hb_logs_param = new HBox();
	//hb_logs_param.setAlignment(Pos.TOP_CENTER);
	hb_logs_param.getChildren().addAll(vb_c, vb_params, vb_cuda);
	//hb_logs_param.setMargin(vb_c, new Insets(0, 0, 0, 0));
        hb_logs_param.setMargin(vb_params, new Insets(0, 64, 0, 64));
        //hb_logs_param.setMargin(vb_cuda, new Insets(0, 0, 0, 0));
	hb_logs_param.setAlignment(Pos.TOP_CENTER);
        hb_logs_param.setStyle("-fx-background-color: #F8F8F8;");

	// layout for center component
	vb_logs_param = new VBox();
	vb_logs_param.getChildren().add(hb_logs_param);
	vb_logs_param.setAlignment(Pos.TOP_CENTER);

        // layout for benchmark screen
        bp_benchmark = new BorderPane();
        bp_benchmark.setLeft(vb_benchmarks);
        bp_benchmark.setCenter(vb_logs_param);
        bp_benchmark.setRight(new Text(" "));   // temp trick for padding on the right
        bp_benchmark.setMargin(vb_title, new Insets(16, 0, 16, 0));
        bp_benchmark.setStyle("-fx-background-color: #F8F8F8;");

        // set the layout to the scene as 800px by 600px
        s_benchmark = new Scene(bp_benchmark, 1280, 720);

        //return bp_benchmark;
    }
    
    /* creates the scene for viewing benchmark results */
    private void createResultsScene() {

        // x-axis
        x_time = new NumberAxis();
        x_time.setLabel("Time (in seconds)");
        
        // y-axis
        y_size = new NumberAxis();
        y_size.setLabel("N elements");
        
        // c/c++ line graph
        xyc_s_c = new XYChart.Series<>();
        xyc_s_c.setName("C/C++");
        
        
        // cuda line graph
        xyc_s_cuda = new XYChart.Series<>();
        xyc_s_cuda.setName("Cuda");
        
        // line chart
        lc_graph = new LineChart<Number, Number>(x_time, y_size);
        lc_graph.getData().addAll(xyc_s_c, xyc_s_cuda);
	lc_graph.setMinHeight(512);
        
        // layout for results screen
        vb_results = new VBox();
	vb_results.setAlignment(Pos.TOP_CENTER);
        //vb_results.setMargin(lc_graph, new Insets(0, 0, 6, 0));
	
        
        // set the layout to the scene as 800px by 600px
        //s_results = new Scene(bp_layout, 800, 600);

        //return vb_results;
    }

    public static void main(String[] args) {
        launch(args);
    }

    public void start(Stage primaryStage) throws Exception {

	System.out.println("==== LAUNCHING ====");

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
        mainStage.setResizable(true);
        mainStage.setOnCloseRequest(e -> safetyExit());
        mainStage.show();
        pf_password.requestFocus();
        
        // notify user of 2FA
        drawDialog("Duo 2FA");
    }
    
    /* called when application closed, ensures all connections terminated before exiting */
    private void safetyExit() {
        
        // if active SSH connection opened, error here
        if (connected) {
            
            // close the opened connection
            disconnect();
        }
        
        Platform.exit();
    }
    
    /* connects to a server on valid text field inputs */
    private void connect() {
        
        try {

            // get username, IP, and port from text fields
            username = tf_username.getText();
	    host = cb_ip.getValue();
            port = Integer.parseInt(tf_port.getText());
            
            remote_shell = new RemoteShell(username, pf_password.getText(), host, port);
            
            // test if successful connection
            if (remote_shell.isConnected()) {
                
                // update connected status
                connected = true;
                
		// reset to default styling
		doLoginFailed(false);

                // set to benchmark scene
                setCurrentScene(1);
            }
            
            // otherwise prompt error
            else {
                doLoginFailed(true);
            }

        } catch (Exception e) {
            System.out.println("> Exception @ connect()\n" + e);
        }
    }
    
    /* disconnect from server and return to login screen */
    private void disconnect() {

        try {
            
            // reset default login inputs
            tf_username.setText(username);
            pf_password.clear();
	    cb_ip.setValue(host);
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
		bp_login.setTop(vb_title);
		bp_login.setCenter(vb_connect);
                bp_login.setBottom(hb_footer);
                inResultsView = false;
                pf_password.requestFocus();
                break;

            // mode 1 -> set to benchmark screen
            case 1:
                title += "Run Benchmarks";
                mainStage.setScene(s_benchmark);
                bp_benchmark.setTop(vb_title);
                bp_benchmark.setCenter(vb_logs_param);
                bp_benchmark.setBottom(hb_footer);
                inResultsView = false;
                break;
                
            // mode 2 -> set to results screen
            case 2:
                title += "Benchmark Results";
                vb_results.getChildren().clear();
                vb_results.getChildren().addAll(lc_graph);
                bp_benchmark.setCenter(vb_results);
                inResultsView = true;
                drawGraph();
                break;
        }

        // set scene title, prevent resizing of window, and show the scene
        mainStage.setTitle(title);
        mainStage.setResizable(false);
        mainStage.show();
    }
    
    /* determines whether the text fields should be highlighted in red */
    private void doLoginFailed(boolean flag) {

	// reset to default styling
	if (flag == false) {
	    cb_ip.setStyle(null);
	    cb_ip.setStyle("-fx-font: 14px \"Monospace\";");
	    tf_port.setStyle(null);
	    tf_username.setStyle(null);
	    pf_password.setStyle(null);
	}

	// otherwise login was failed
	else {
	    cb_ip.setStyle("-fx-font: 14px \"Monospace\";-fx-border-color: red;");
	    tf_port.setStyle("-fx-text-box-border: red;");
	    tf_username.setStyle("-fx-text-box-border: red;");
	    pf_password.setStyle("-fx-text-box-border: red;");
	}
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
            rb.setStyle("-fx-border-style: solid;"
                  + "-fx-border-width: 0 0 2 0;"
                  + "-fx-padding: -16 48 16 0;"
                  + "-fx-border-color: #d8d8d8;");
        }
    }


    /* draws the notification box to the active window */
    private void drawDialog(String type) {

	// only show when not already showing
	if (hasNotification == false) {

	    // determine which dialog box is created	
	    switch (type) {

		// about info
		case "About":
		    pd_message = new PageDialog(type, new String[]{"This is the about. Currently this is not filled out but the idea is that there will be helpful stuff here. Any thoughts?\nLine 2\nLine 3"});
		    break;

		// contact info
		case "Contact":
		    pd_message = new PageDialog(type, new String[]{"Contact us at:\n\tkkowal28@uic.edu\n\tacer.uic.edu"});
		    break;

		// help info
		case "Help":
		    pd_message = new PageDialog(type, new String[]{"This is a help dialog. Currently this is not filled out but the idea is that there will be helpful stuff here. Any thoughts?\nLine 2\nLine 3", "A second page."});
		    break;
		
                // help info
		case "Duo 2FA":
		    pd_message = new PageDialog(type, new String[]{"These servers use Duo Two Factor Authentication. By default your first Duo option has been selected. Please verify\nyour log-in. \n\n(Verification is often times a push notification or SMS.)"});
		    break;
            }

	    // determine which scene to place notification
	    if (!connected) {
		vb_connect.getChildren().add(pd_message);
		vb_connect.setMargin(pd_message, new Insets(64, 0, 0, 0));
	    }
	    else {
		// if at benchmark scene
		if (inResultsView == false) {
		    vb_logs_param.getChildren().add(pd_message);
		    vb_logs_param.setMargin(pd_message, new Insets(64, 0, 0, 0));
		}
		
		// otherwise at results view
		else {
		    vb_results.getChildren().add(pd_message);
		    vb_results.setMargin(pd_message, new Insets(32, 0, 0, 0));
		}
 	    }

	    hasNotification = true;
	}
    }

    /* removes the notification box form the active window */
    public static void removeDialog() {
	vb_connect.getChildren().remove(pd_message);
	vb_logs_param.getChildren().remove(pd_message);
	vb_results.getChildren().remove(pd_message);
	hasNotification = false;
    }
    
    /* wrapper for appending to text area logs */
    public static void appendToLog(String type, String s) {
	
	// determine which log to append to
	if (type.equals("c_cpp"))
            ta_log_c.appendText(s + "\n");
	else
	    ta_log_cuda.appendText(s + "\n");
    }
    
    /* button handler for results/benchmark scene switching */
    private void benchmarkSceneSwitch() {
        
        // if at benchmark scene
        if (!inResultsView) {
            setCurrentScene(2);
            bp_benchmark.setCenter(vb_results);
            b_results.setText("Benchmarks");
        }
        
        // otherwise at graph scene
        else {
            setCurrentScene(1);
            bp_benchmark.setCenter(vb_logs_param);
            b_results.setText("Results");
        }
    }
    
    /* button listener for benchmark radio buttons */
    private void benchmarkButtonListener() {
        
        // if at benchmark scene
        if (!inResultsView) {

	    String current_benchmark = getCurrentBenchmark();

            appendToLog("c_cpp", "> Selected " + current_benchmark + " benchmark");
	    appendToLog("cuda", "> Selected " + current_benchmark + " benchmark");

	    // lock certain oarameters based on current benchmark selection
	    //cb_size.setDisable(false);
	    //cb_runs.setDisable(false);

	    switch (current_benchmark) {
		case "matrix_mult":
		    //cb_runs.setDisable(true);
		    break;
	    }

        }
        
        // otherwise at graph scene
        else {
            drawGraph();
        }
    }
    
    /* fetches the currently selected benchmark */
    private String getCurrentBenchmark() {
        
        // get current radio button selection
        RadioButton rb = (RadioButton)tg_benchmarks.getSelectedToggle();
        
        return rb.getText().trim();
    }
    
    /* fetches the currently selected benchmark's ID */
    private String[] getBenchmarkInfo() {
        
        // store benchmark info as name and parameters
        String benchmarkInfo[] = {"benchmark", "size/runs", "threads"};
        
        // get selected benchmark information
        switch (getCurrentBenchmark()) {
            case "Matrix Multiplication":
                benchmarkInfo[0] = "matrix_mult";
                benchmarkInfo[1] = "-n " + cb_size.getValue();
                benchmarkInfo[2] = " -t " + cb_threads.getValue();
                break;
            case "Single-Precision AX+Y":
                benchmarkInfo[0] = "saxpy";
                benchmarkInfo[1] = "-r " + cb_runs.getValue();
                benchmarkInfo[2] = " -t " + cb_threads.getValue();
                break;
            case "Vector Addition":
                benchmarkInfo[0] = "vector_add";
                break;
            case "#test_4":
                benchmarkInfo[0] = "t4";
                break;
        }
        
        return benchmarkInfo;
    }
    
    /* runs the specified benchmark */
    private void runBenchmark(String type) {
        
        // run benchmark given code type, benchmark name, and custom parameter
	try {
            remote_shell.runBenchmark(type, getBenchmarkInfo());
        } catch (JSchException ex) {
            Logger.getLogger(ACER_Benchmark.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("> JSch Error @ runBenchmark()");
        } catch (IOException ex) {
            Logger.getLogger(ACER_Benchmark.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("> I/O Error @ runBenchmark()");
        }     
    }
    
    /* writes the data to the appropriate file */
    public static void writeResult(String type, String benchmark, String data) {
        
        String file = benchmark;
        
        switch (type) {
            case "c_cpp":
                file += "_c_cpp.txt";
                break;
            case "cuda":
                file += "_cuda.txt";
                break;
        }
        
        // get correct path to results file
        Path path = Paths.get("src/main/resources/results/", file);
        
        try {
            Files.write(path, data.getBytes(), StandardOpenOption.APPEND);
        }
        catch (Exception e) {
            System.out.println("> Error writing to file\n" + e);
        }
    }
    
    /* draws a graph based on the data from the current benchmark selection */
    private void drawGraph() {
        
        // rename graph
        lc_graph.setTitle(getCurrentBenchmark());
        
        // clear old graph
        xyc_s_c.getData().clear();
        xyc_s_cuda.getData().clear();
        
        // get correct path to benchmark's c/c++ results file
        Path path = Paths.get("src/main/resources/results/" + getBenchmarkInfo()[0] + "_c_cpp.txt");
        
        try {
            List<String> list = Files.readAllLines(path);
            
            // for each line in file with data, add to graph
            list.forEach(line -> {
                if (line.charAt(0) != '#') {
                    
                    // extract time and size/runs data point
                    Double time = Double.parseDouble(line.substring(0, line.indexOf(',')));
                    Integer parameter = Integer.parseInt(line.substring(line.indexOf(',') + 1));
                    
                    // add data to graph
                    xyc_s_c.getData().add(new XYChart.Data(time, parameter, "08/07/2019"));
                }
            });
        }
        catch (Exception e) {
            System.out.println("> Error reading from file\n" + e);
        }
        
        
        // get correct path to benchmark's cuda results file
        path = Paths.get("src/main/resources/results/" + getBenchmarkInfo()[0] + "_cuda.txt");
        
        try {
            List<String> list = Files.readAllLines(path);
            
            // for each line in file with data, add to graph
            list.forEach(line -> {
                if (line.charAt(0) != '#') {
                    
                    // extract time and size data point
                    Double time = Double.parseDouble(line.substring(0, line.indexOf(',')));
                    Integer parameter = Integer.parseInt(line.substring(line.indexOf(',') + 1));
                    
                    // add data to graph
                    xyc_s_cuda.getData().add(new XYChart.Data(time, parameter, "08/07/2019"));
                }
            });
        }
        catch (Exception e) {
            System.out.println("> Error reading from file\n" + e);
        }
        
        // iterate over each series in the graph
        for (final XYChart.Series<Number, Number> series : lc_graph.getData()) {
            
            // iterate over each data point in the current series
            for (final Data<Number, Number> data : series.getData()) {
                
                // create a new tooltip for current data point
                Tooltip t = new Tooltip("Time = \t" + data.getXValue().toString() + "\nSize = \t" + data.getYValue().toString() + "\n\nRan on " + data.getExtraValue());
                Tooltip.install(data.getNode(), t);
                
                // add styling to current data point
                data.getNode().setOnMouseEntered(e -> data.getNode().setStyle("-fx-background-color: white; -fx-background-radius: 8;"));
                data.getNode().setOnMouseExited(e -> data.getNode().setStyle(null));
            }
        }

    }
}
