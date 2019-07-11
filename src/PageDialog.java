package acer_benchmark;

import javafx.scene.control.ButtonType;
import javafx.scene.control.Dialog;
import javafx.scene.layout.GridPane;
import javafx.scene.text.Text;

public class PageDialog extends Dialog {
    
    private String title;
    private String[][] content;
    private int index;
    private int pages;
    
    private ButtonType bt_go;
    
    private Text t_message;
    private Text t_pages;
    
    /* constructor for paged dialog box */
    public PageDialog(int index, String title, String[][] content) {
        this.title = title;
        this.content = content;
        
        this.index = index;
        pages = content.length;
        
        // if not at last page, set button as NEXT
        if (this.index != pages - 1) {
            bt_go = ButtonType.NEXT;
        }
        
        // otherwise at last page so set button as FINISH
        else {
            bt_go = ButtonType.FINISH;
        }
        
        // update the content and page numbers
        t_message = new Text(this.content[this.index][1]);
        t_pages = new Text("\n" + (this.index + 1)+ "/" + pages);
        
        // set the title and header
        setTitle(this.title);
        setHeaderText(this.content[this.index][0]);
        
        // create the layout
        GridPane gp_layout = new GridPane();
        gp_layout.add(t_message, 1, 1);
        gp_layout.add(t_pages, 3, 3);
        
        // set size, layout, and buttons
        getDialogPane().setPrefSize(128, 96);
        getDialogPane().setContent(gp_layout);
        getDialogPane().getButtonTypes().addAll(ButtonType.CANCEL, bt_go);
        
        showAndWait();
        
        // if user clicked NEXT
        if (getResult() == ButtonType.NEXT) {
            
            // spawn the next page
            new PageDialog(++this.index, this.title, this.content);
        }
    }
}
