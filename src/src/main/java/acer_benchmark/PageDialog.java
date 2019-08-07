package acer_benchmark;

import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.Button;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.text.Text;

public class PageDialog extends GridPane {

    private final String title;
    private final String[] content;
    private int index;
    private final int pages;
    
    private Button b_go;
    private Text t_title, t_pages, t_content;
    private HBox hb_title, hb_blank1, hb_blank2, hb_content, hb_pages, hb_footer;
    
    /* constructor for page dialog box */
    public PageDialog(String title, String[] content) {
        this.title = title;
        this.content = content;
	this.index = 0;
	this.pages = content.length;
        
	// update the content and page number
	t_title = new Text(title);
	t_title.setStyle("-fx-font: 18 Arial;");
	t_pages = new Text("" + (this.index + 1) + "/" + pages);
	t_pages.setStyle("-fx-font: 18 Arial;");
	t_content = new Text(content[this.index]);

	// set up cells of layout
	hb_title = new HBox(t_title);
	hb_title.setAlignment(Pos.BOTTOM_LEFT);
	hb_pages = new HBox(t_pages);
	hb_pages.setAlignment(Pos.BOTTOM_RIGHT);
	hb_blank1 = new HBox();
	hb_blank1.setStyle("-fx-border-style: solid;"
                         + "-fx-border-width: 3 0 3 0;"
                         //+ "-fx-padding: 0 48 8 0;"
                         + "-fx-border-color: black;");
	hb_content = new HBox(t_content);
	hb_content.setAlignment(Pos.CENTER_LEFT);
	hb_content.setMinSize(596, 64);
	hb_content.setMaxSize(596, 64);
	hb_content.setStyle("-fx-border-style: solid;"
                         + "-fx-border-width: 3 0 3 0;"
                         //+ "-fx-padding: 0 48 8 0;"
                         + "-fx-border-color: black;");
	hb_blank2 = new HBox();
	hb_blank2.setStyle("-fx-border-style: solid;"
                         + "-fx-border-width: 3 0 3 0;"
                         //+ "-fx-padding: 0 48 8 0;"
                         + "-fx-border-color: black;");

        // if not at last page, set button as NEXT
        if (this.index != pages - 1) {
	    b_go = new Button("Next");
        }
        
        // otherwise at last page so set button as FINISH
        else {
	    b_go = new Button(" OK ");
        }

	// set up footer
	hb_footer = new HBox(b_go);
	hb_footer.setAlignment(Pos.BOTTOM_RIGHT);
        
	// set up button handler
	b_go.setMinWidth(64);
	b_go.setOnAction(e -> buttonHandler());
	resetButtonStyle(b_go, "#648000");
	b_go.setOnMouseEntered(e -> setButtonHoverStyle(b_go, "#4b6000"));
        b_go.setOnMousePressed(e -> setButtonPressedStyle(b_go, "#4b6000"));
        b_go.setOnMouseExited(e -> resetButtonStyle(b_go, "#648000"));

	this.add(hb_title, 0, 0);
	this.add(hb_pages, 2, 0);
	this.add(hb_blank1, 0, 1);
	this.add(hb_content, 1, 1);
	this.add(hb_blank2, 2, 1);
	this.add(b_go, 2, 2);
	this.setMargin(hb_title, new Insets(0, 0, 0, 2));
	this.setMargin(hb_pages, new Insets(0, 2, 0, 0));
	this.setMargin(hb_blank1, new Insets(8, 0, 8, 0));
	this.setMargin(hb_content, new Insets(8, 0, 8, 0));
	this.setMargin(hb_blank2, new Insets(8, 0, 8, 0));
        this.setMargin(b_go, new Insets(4, 4, 0, 0));
	this.setAlignment(Pos.CENTER);
	this.setMaxSize(728, 148);
	this.setMinSize(728, 148);
	this.setStyle("-fx-background-color: rgba(200, 200, 200, 0.15);");
    }

    /* event handler for button press */
    private void buttonHandler() {

	// update next page if available
	if (b_go.getText() == "Next") {
	    updatePage();
	}

	// end of dialog reached
	else {
	    //this.getChildren().clear();
	    ACER_Benchmark.removeDialog();
	}
    }

    /* resets the specified button's style */
    private void resetButtonStyle(Button b, String hex_color) {
        b.setStyle("-fx-background-color: " + hex_color + ";"
                + "-fx-text-fill: black;"
                + "-fx-font-weight: bold;"
                + "-fx-font-family: Calibri;"
                + "-fx-font-size: 14px;");
    }
    
    /* adjusts the specified button's style for hover event */
    private void setButtonHoverStyle(Button b, String hex_color) {
        b.setStyle("-fx-background-color: " + hex_color + ";"
                + "-fx-text-fill: black;"
                + "-fx-font-weight: bold;"
                + "-fx-font-family: Calibri;"
                + "-fx-font-size: 14px;");
    }
    
    /* adjusts the specified button's style for press event */
    private void setButtonPressedStyle(Button b, String hex_color) {
        b.setStyle("-fx-background-color: " + hex_color + ";"
                + "-fx-text-fill: black;"
                + "-fx-font-weight: bold;"
                + "-fx-font-family: Calibri;"
                + "-fx-font-size: 14px;");
    }
    /* update the content, page, and button text */
    private void updatePage() {
	
	// update dialog content
	t_pages.setText("" + (++index + 1) + "/" + pages);
	t_content.setText(content[index]);

	// if at last page, set button as FINISHED
        if (this.index == pages - 1) {
            b_go.setText(" OK ");
        }
    }
}
