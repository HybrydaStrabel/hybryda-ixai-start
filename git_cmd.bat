echo "Sprawdzanie statusu repozytorium..."
git status

echo "Dodawanie zmian..."
git add .

echo "Tworzenie nowego commitu..."
git commit -m "Dodano nowe wersje"

echo "Pobieranie najnowszych zmian z GitHub..."
git pull origin main --rebase

echo "Wysylanie zmian do repozytorium..."
git push origin main

echo "Ostateczny status po wykonaniu operacji:"
git status

echo "Sprawdzanie róznic miedzy lokalnym repo a GitHub..."
git fetch origin main
git diff origin/main

echo "Jesli powyzej nie ma zadnego wyniku, repozytoria sa identyczne."