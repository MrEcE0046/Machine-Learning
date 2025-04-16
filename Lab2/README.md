## Rekomendationsystem för filmer

Programmet rekommenderar 5 filmer från en given film genom TF-IDF vektorisering och Nearest neighbors. 

Programmet läser in filerna med taggar, filmer och ratings. Då jag ville utforska ord hantering valde jag att göra en TF-IDF vektorisering på genre och tags. TF-IDF vektorn väger hur viktigt ett ord är för en film i en samling. Många användare har sina unika taggar för samma film så alla tags samlades för varje film till en cell. Så istället för hundratals rader med tags för samma film är det nu en film per rad med alla dess tags samlade till den. Både genre och tags städas enhetligt. 
Genre och tags vektoriseras separat och sätts sedan ihop horisontalt till en kombinerad matris. Nu har varje film en rad som representerar både tags och genre i en enda vektor. 

Matrisen tränar en Nearest neighbor modell som hittar närmaste grannar baserat på cosinuslikhet. 

Rekomendationen görs genom att hämta den givna filmens vektor och använda en obevakad kneighbors funktion som hittar de närmaste filmerna baserat på innehållet och därefter returnerar filmerna. 

Här var programmet slut till en början men jag ville utforska på gott och ont om man kunde addera information för att få bättre rekommendationer. 
Målet var att få in betygen från alla användare, men att använda hela datan var väldigt beräkningstung. Datan innehåller ca 35000 olika titlar, så önskan var att få ned den summan till en mer hanterbar summa. Plananen blev att öka antal rekommendationer från TF-IDF vektoriseringen från 5 till 50 filmer och beräkna dem, vilket gör uträkningen billigare. 

Ratings skalades ned genom att ta bort användare som hade mindre än 200 kommentarer och skapade data på hur många gånger en film fått en rating för att sedan sammafogas med de 50 titlar som den första gallringen av titlar sker. Detta görs om varje gång man väljer en ny titel.

De 50 titlar på givet en titel har nu metadata för titlar och ratings. Med dem skapas en pivottabell. Med risk att pivottabellen skulle bli för tung konverterades tabellen till en gles matris (csr_matris) för minnesoptimering. Den går igenom nearestneighbors funktion och tränar modellen. 

En ny rekommendation görs av en liknande obevakad KNN baserad likhetsjämnförelse med andra filmer och returnerar de 5 närmaste grannarna. 

Tanken var att skapa en streamlit applikation och med en API nyckel jag har hämtat för att ladda titlebilder från tmdb och presentera 5 filmer. Dock när applikationen skulle laddas upp visade det sig att en av filerna var för stor så för att undvika att lägga dyrbar tid på outputen skapades en sista funktion där man interagerar med appen genom terminalen. 

Applikationen är således oehört användar ovänlig men så länge man håller sig till reglerna spottar den ut bra rekommendationer. Ingen tid har alltså spenderats på användar vänligheten, tyvärr. 
För att underlätta för användaren ger programmet användaren 10 förslag som kan copy/paste i sökraden. Men har man tillgång till hela datasetet kan man kopiera vilken titel som helst. Det är titlel och årtal inom parates som accepteras av programmet. 
Tex Toy Story (1995) eller Heat (1995).