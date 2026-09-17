import com.integrallis.models.runtime.chat.*;
import java.util.List;
public class Render {
  public static void main(String[] a) {
    for (ChatTemplate t : ChatTemplate.values()) {
      try {
        var p = t.render(List.of(ChatMessage.user("UQ"), ChatMessage.assistant("ANSWERX"), ChatMessage.user("U2")));
        String text = p.text(); int i = text.indexOf("ANSWERX");
        String after = i < 0 ? "<assistant text not rendered>" : text.substring(i + 7, Math.min(text.length(), i + 7 + 40));
        System.out.println(t.id() + " => " + after.replace("\n", "\\n"));
      } catch (Exception e) { System.out.println(t.id() + " !! " + e.getMessage()); }
    }
  }
}
