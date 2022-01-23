#include "Header.h"

#include "CServerManager.h"

int __cdecl main()
{
	//std::unique_ptr<CServerManager> cServerManger(new CServerManager());

	CServerManager server;

	server.Initialize();
	
	// ������ ����
	std::thread t1{ &CServerManager::ControllerProcess, &server };
	std::thread t2{ &CServerManager::JetBotProcess,		&server };
	
	// �����尡 ������ �� ���� ���
	t1.join();
	t2.join();

	server.Release();

	return 0;
}