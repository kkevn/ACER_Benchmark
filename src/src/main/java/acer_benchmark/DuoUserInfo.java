package acer_benchmark;

import com.jcraft.jsch.UIKeyboardInteractive;
import com.jcraft.jsch.UserInfo;
public class DuoUserInfo implements UserInfo, UIKeyboardInteractive {

    private String password;
    
    /* constructor to store password parameter */
    DuoUserInfo(String pass) {
        password = pass;
    }

    /* allow RSA key fingerprint */
    public boolean promptYesNo(String str) {
        return true;
    }

    /* display Duo registration message */
    public void showMessage(String message) {
        System.out.println(message);
    }

    /* unused */
    public String getPassphrase() {
        return null;
    }

    /* unused */
    public boolean promptPassphrase(String message) {
        return false;
    }

    /* return password passed through constructor */
    public String getPassword() {
        return password;
    }

    /* password is required so must be prompted */
    public boolean promptPassword(String message) {
        return false;
    }

    /* achieve user interaction for Duo 2FA */
    public String[] promptKeyboardInteractive(String destination, String name, String instruction, String[] prompt, boolean[] echo) {

        System.out.println("destination: " + destination);
        System.out.println("name: " + name);
        System.out.println("instruction: " + instruction);
        System.out.println("prompt.length: " + prompt.length);

        String[] str = new String[1];

        // output the prompts to console
        for (int i = 0; i < prompt.length; i++) {
            System.out.println("\n" + prompt[i] + "\n");
        }
        
        // automatically send inputs when prompted
        if (prompt[0].contains("Password:")) {
            str[0] = getPassword();
        } else if (prompt[0].contains("Passcode")) {
            str[0] = "1";   // use first Duo option by default
        } else {
            str = null;
        }
        return str;
    }
}
