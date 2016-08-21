/////////////////////////////////////////////////////////////////////////////////////////
//	Author: Brian Beallo                                                               //
//                                                                                     //
//  Description: A genetic algorithm to solve for where you can place queens           //
//                 on a chess board so that no two queens attack each other            //
//                                                                                     //
//                                                                                     //
//                                                                                     //
/////////////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <vector>
#include <string>
#include <time.h>
#include <math.h>
#include <iomanip>
#include <fstream>

using namespace std;

int main()
{
	//initialize Variables
	fstream fout("Board.txt");
	int size, board_number, r, t, u, s, solvedboard, loops, time_min, time_hour;
	string run_check;
	double time1, time2, time_sec;
	bool run = 0, loop = false;
	vector<vector<vector<int>>> Boards;
	vector<vector<vector<int>>> Sums;
	vector<vector<int>> hits;
	//Main loop
	while (run == 0)
	{
		//Initialize random seed
		srand(int(time(NULL)));
		solvedboard = -1;
		//Clear screen
		system("CLS");
		//Get Board Size and Verify input
		do
		{
			loop = false;
			if (cin.fail())
			{
				cout << "************Not a number**************\n\n";
				cin.clear();
				cin.ignore(500, '\n');
			}

			cout << "Input board size (Must be greater than 3)\n\t:";
			cin >> size;

			if (size <= 3 && !cin.fail())
			{
				loop = true;
				cout << "Must be within range (greater than 3)\n\n";
			}

		} while (cin.fail() || loop == true);

		//Get Number of test boards and Verify input
		do
		{
			loop = false;
			if (cin.fail())
			{
				cout << "************Not a number**************\n\n";
				cin.clear();
				cin.ignore(500, '\n');
			}

			cout << "input number of boards to test with (at least 10)\n\t:";
			cin >> board_number;

			if (board_number < 10 && !cin.fail())
			{
				loop = true;
				cout << "Must be within range (at least 10)\n\n";
			}

		} while (cin.fail() || loop == true);

		cout << "Running\n\n";
		time1 = clock();
		//Calculate the Solution
		//Create Pair boards
		vector<int> Pairs(board_number, 0);
		//Create and Fill 3d vector array with boards of the given size with 0's
		Boards.resize(board_number);
		for (int i = 0; i < board_number; i++)
		{
			Boards.at(i).resize(size);
		}

		for (int i = 0; i < board_number; i++)
		{
			for (int j = 0; j < size; j++)
			{
				Boards.at(i).at(j).resize(size + 1);
				for (int k = 0; k < size + 1; k++)
				{
					Boards.at(i).at(j).at(k) = 0;
				}
			}
		}

		//Generate random Boards
		for (int i = 0; i < board_number; i++)
		{
			for (int j = 0; j < size; j++)
			{
				r = rand() % size;
				Boards.at(i).at(j).at(size) = r;
				Boards.at(i).at(j).at(r) = 1;
			}
		}

		//Solve loop
		loops = 0;
		while (solvedboard == -1)
		{
			loops++;
			//Calculations: Setup
			int bestboard0 = rand() % board_number, bestboard1 = rand() % board_number;
			Sums.resize(3);
			Sums.at(0).resize(board_number);
			Sums.at(1).resize(board_number);
			Sums.at(2).resize(board_number);
			for (int i = 0; i < board_number; i++)
			{
				Sums.at(0).at(i).resize(size);
				for (int j = 0; j < size; j++)
				{
					Sums.at(0).at(i).at(j) = 0;
				}
				Sums.at(1).at(i).resize(2 * size - 1);
				for (int j = 0; j < 2 * size - 1; j++)
				{
					Sums.at(1).at(i).at(j) = 0;
				}
				Sums.at(2).at(i).resize(2 * size - 1);
				for (int j = 0; j < 2 * size - 1; j++)
				{
					Sums.at(2).at(i).at(j) = 0;
				}
			}

			//Calculations; Count Vertical
			for (int i = 0; i < board_number; i++)
			{
				for (int j = 0; j < size; j++)
				{
					for (int k = 0; k < size; k++)
					{
						Sums.at(0).at(i).at(j) += Boards.at(i).at(k).at(j);
					}
				}
			}

			//Calculations: Count Diagonal UL->DR
			for (int i = 0; i < board_number; i++)
			{
				for (int j = 0; j < size; j++)
				{
					for (int k = 0; k < size; k++)
					{
						Sums.at(1).at(i).at(k - j + (size - 1)) += Boards.at(i).at(j).at(k);
					}
				}
			}

			//Calculations: Count Diagonal DL->UR
			for (int i = 0; i < board_number; i++)
			{
				for (int j = 0; j < size; j++)
				{
					for (int k = 0; k < size; k++)
					{
						Sums.at(2).at(i).at(k + j) += Boards.at(i).at(j).at(k);
					}
				}
			}

			//Calculations: Count Queen Pairs
			for (int i = 0; i < board_number; i++)
			{
				Pairs.at(i) = 0;
				for (int j = 0; j < size; j++)
				{
					for (int k = Sums.at(0).at(i).at(j) - 1; k > -1; k--)
					{
						Pairs.at(i) += k;
					}
				}
				for (int j = 0; j < 2 * size - 1; j++)
				{
					for (int k = Sums.at(1).at(i).at(j) - 1; k > -1; k--)
					{
						Pairs.at(i) += k;
					}
				}
				for (int j = 0; j < 2 * size - 1; j++){
					for (int k = Sums.at(2).at(i).at(j) - 1; k > -1; k--)
					{
						Pairs.at(i) += k;
					}
				}
			}

			//Select 2 Best Boards
			for (int i = 0; i < board_number; i++)
			{
				for (int j = 0; j < board_number; j++)
				{
					if (Pairs.at(j) < Pairs.at(bestboard0) && j != bestboard0 && j != bestboard1)
					{
						bestboard0 = j;
					}
					if (Pairs.at(j) < Pairs.at(bestboard1) && j != bestboard0 && j != bestboard1)
					{
						bestboard1 = j;
					}
				}
			}

			//Select Solved Board and exit loop
			if (Pairs.at(bestboard0) == 0)
			{
				solvedboard = bestboard0;
			}
			if (Pairs.at(bestboard1) == 0)
			{
				solvedboard = bestboard1;
			}

			if (loops % 100 == 0)
			{
				cout << "Current: " << loops << ", " << Pairs.at(bestboard0) << ", " << Pairs.at(bestboard1) << endl;
			}

			//If not solved
			if (solvedboard == -1)
			{
				//Count individual Queen hits
				hits.resize(2);
				for (int i = 0; i < 2; i++)
				{
					hits.at(i).resize(size);
				}
				for (int j = 0; j < size; j++)
				{
					hits.at(0).at(j) = -3;
					hits.at(0).at(j) += Sums.at(0).at(bestboard0).at(Boards.at(bestboard0).at(j).at(size));
					hits.at(0).at(j) += Sums.at(1).at(bestboard0).at(Boards.at(bestboard0).at(j).at(size) - j + size - 1);
					hits.at(0).at(j) += Sums.at(2).at(bestboard0).at(Boards.at(bestboard0).at(j).at(size) + j);
				}
				for (int j = 0; j < size; j++)
				{
					hits.at(1).at(j) = -3;
					hits.at(1).at(j) += Sums.at(0).at(bestboard1).at(Boards.at(bestboard1).at(j).at(size));
					hits.at(1).at(j) += Sums.at(1).at(bestboard1).at(Boards.at(bestboard1).at(j).at(size) - j + size - 1);
					hits.at(1).at(j) += Sums.at(2).at(bestboard1).at(Boards.at(bestboard1).at(j).at(size) + j);
				}

				//Erase Boards
				for (int i = 0; i < board_number; i++)
				{
					if (i != bestboard0 && i != bestboard1)
					{
						for (int j = 0; j < size; j++)
						{
							for (int k = 0; k < size + 1; k++)
							{
								Boards.at(i).at(j).at(k) = 0;
							}
						}
					}
				}

				//Create Child Boards
				for (int j = 0; j < board_number; j++)
				{
					if (j != bestboard0 && j != bestboard1)
					{
						for (int i = 0; i < size; i++)
						{
							r = rand() % 2;
							t = (rand() % 10);
							u = (rand() % 10);

							if (hits.at(0).at(i) < hits.at(1).at(i))
							{
								if (u == 0)
								{
									s = rand() % size;
									Boards.at(j).at(i).at(size) = s;
									Boards.at(j).at(i).at(s) = 1;
								}
								else
								{
									Boards.at(j).at(i).at(size) = Boards.at(bestboard0).at(i).at(size);
									Boards.at(j).at(i).at(Boards.at(bestboard0).at(i).at(size)) = 1;
								}
							}
							if (hits.at(1).at(i) < hits.at(0).at(i))
							{

								if (u == 0)
								{
									s = rand() % size;
									Boards.at(j).at(i).at(size) = s;
									Boards.at(j).at(i).at(s) = 1;
								}
								else
								{
									Boards.at(j).at(i).at(size) = Boards.at(bestboard1).at(i).at(size);
									Boards.at(j).at(i).at(Boards.at(bestboard1).at(i).at(size)) = 1;
								}
							}
							if (hits.at(0).at(i) == hits.at(1).at(i))
							{
								if (r == 0)
								{
									if (t == 0)
									{
										s = rand() % size;
										Boards.at(j).at(i).at(size) = s;
										Boards.at(j).at(i).at(s) = 1;
									}
									else
									{
										Boards.at(j).at(i).at(size) = Boards.at(bestboard0).at(i).at(size);
										Boards.at(j).at(i).at(Boards.at(bestboard0).at(i).at(size)) = 1;
									}
								}
								if (r == 1)
								{
									if (t == 0)
									{
										s = rand() % size;
										Boards.at(j).at(i).at(size) = s;
										Boards.at(j).at(i).at(s) = 1;
									}
									else
									{
										Boards.at(j).at(i).at(size) = Boards.at(bestboard1).at(i).at(size);
										Boards.at(j).at(i).at(Boards.at(bestboard1).at(i).at(size)) = 1;
									}
								}
							}
						}
					}
				}
			}
		}
		time2 = clock();
		time_sec = (time2 - time1) / CLOCKS_PER_SEC;
		time_min = int(time_sec / 60);
		time_hour = (time_min / 60);
		time_min = time_min % 60;
		time_sec -= (double(time_min) * 60 + double(time_hour) * 3600);

		//Print Solved Board
		fout.open("Board.txt");
		cout << fout.is_open() << endl;
		for (int i = 0; i < size; i++)
		{
			cout << "|";
			fout << "|";
			for (int j = 0; j < size; j++)
			{
				if (Boards.at(solvedboard).at(i).at(j) == 0)
				{
					cout << "-|";
					fout << "-|";
				}
				if (Boards.at(solvedboard).at(i).at(j) == 1)
				{
					cout << "Q|";
					fout << "Q|";
				}
			}
			cout << "\n";
			fout << "\n";
		}
		cout << endl << loops << endl;
		fout << endl << loops << endl;
		cout << "Hour: " << time_hour << " Min: " << time_min << " Sec: " << fixed << setprecision(2) << time_sec << endl;
		fout << "Hour: " << time_hour << " Min: " << time_min << " Sec: " << fixed << setprecision(2) << time_sec << endl;
		fout.close();
		//Repeat Program
		while (1 == 1)
		{
			cout << "Repeat? (Y/N):";
			cin >> run_check;
			if (run_check == "N" || run_check == "n")
			{
				run = 1;
				break;
			}
			if (run_check == "Y" || run_check == "y")
			{
				break;
			}
			else
			{
				cout << "Incorrect Input, Try Again" << endl;
			}
		}
	}

	cout << endl;
	system("pause");
	return 0;
}